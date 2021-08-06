import torchtext.data as data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import numpy as np
from collections import defaultdict
import time
import math
from app.seq2seq.dataset import Seq2SeqDataset
import sys
import os
from app.nlp.tokenizer import get_tokenizer


class FeedForwardLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class MultiHeadedAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, n_attention_heads, dropout):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // n_attention_heads
        self.n_attention_heads = n_attention_heads

        self.fc_q = nn.Linear(embedding_dim, embedding_dim)
        self.fc_k = nn.Linear(embedding_dim, embedding_dim)
        self.fc_v = nn.Linear(embedding_dim, embedding_dim)

        self.fc_o = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(p=dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.n_attention_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_attention_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_attention_heads,
                   self.head_dim).permute(0, 2, 1, 3)

        # TODO maybe use batch matrix multiplication
        score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(score, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.embedding_dim)

        x = self.fc_o(x)
        return x, attention


class EncoderLayer(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, n_attention_heads, hidden_dim, dropout):
        super().__init__()
        self.attention_layer_norm = nn.LayerNorm(embedding_dim)
        self.fc_layer_norm = nn.LayerNorm(embedding_dim)
        self.self_attention = MultiHeadedAttentionLayer(
            embedding_dim, n_attention_heads, dropout)
        self.fc = FeedForwardLayer(embedding_dim, hidden_dim, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)

        # residual connection
        src = self.attention_layer_norm(src + self.dropout(_src))

        _src = self.fc(src)
        src = self.fc_layer_norm(src+self.dropout(_src))
        return src


class Encoder(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, n_layers, n_attention_heads, hidden_dim, dropout, max_sequence_length=100):
        super().__init__()

        self.word_embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        self.positional_embedding = nn.Embedding(
            max_sequence_length, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(
            vocabulary_size, embedding_dim, n_attention_heads, hidden_dim, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(p=dropout)
        self.scale = torch.sqrt(torch.FloatTensor([embedding_dim]))

    def forward(self, src, src_mask):
        # size(src)= BATCH_SIZE x SEQUNCE_LENGTH
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # Next thing to do is to create positions for src_len that can be consumed by the positional embedding
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1)

        src = self.dropout((self.word_embeddings(
            src) * self.scale) + self.positional_embedding(pos))
        # size(src) = BATCH_SIZExSEQUENCE_LENGTH_EMBEDDING_DIM

        for layer in self.layers:
            src = layer(src, src_mask)

        return src


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_attention_heads, hidden_dim, dropout):
        super().__init__()
        self.self_attention_layer_norm = nn.LayerNorm(embedding_dim)
        self.encoder_attention_layer_norm = nn.LayerNorm(embedding_dim)
        self.fc_layer_norm = nn.LayerNorm(embedding_dim)

        self.self_attention = MultiHeadedAttentionLayer(
            embedding_dim, n_attention_heads, dropout)
        self.encoder_attention = MultiHeadedAttentionLayer(
            embedding_dim, n_attention_heads, dropout)
        self.fc = FeedForwardLayer(embedding_dim, hidden_dim, dropout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, trg, encoder_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attention_layer_norm(trg+self.dropout(_trg))

        _trg, attention = self.encoder_attention(
            trg, encoder_src, encoder_src, src_mask)
        trg = self.encoder_attention_layer_norm(trg+self.dropout(_trg))

        _trg = self.fc(trg)
        trg = self.fc_layer_norm(trg+self.dropout(_trg))

        return trg, attention


class Decoder(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, n_layers, n_attention_heads, hidden_dim, dropout, max_length=100):
        super().__init__()

        self.word_embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        self.positional_embeddings = nn.Embedding(max_length, embedding_dim)
        self.layers = nn.ModuleList([DecoderLayer(
            embedding_dim, n_attention_heads, hidden_dim, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(embedding_dim, vocabulary_size)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = torch.sqrt(torch.FloatTensor([embedding_dim]))

    def forward(self, target, encoder_src, target_mask, src_mask):
        batch_size = target.shape[0]
        target_length = target.shape[1]

        pos = torch.arange(0, target_length).unsqueeze(0).repeat(batch_size, 1)

        trg = self.dropout((self.word_embeddings(target) *
                            self.scale) + self.positional_embeddings(pos))
        # size(src) = BATCH_SIZExSEQUENCE_LENGTH_EMBEDDING_DIM

        for layer in self.layers:
            trg, attention = layer(trg, encoder_src, target_mask, src_mask)
        output = self.fc_out(trg)
        return output, attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_padding_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_padding_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention


def init_model(fileName):
    dir_path = "datasets/"
    dialog_path = os.path.join(dir_path, fileName)
    t = Seq2SeqDataset(dialog_path, tokenize=get_tokenizer)

    SRC, TARGET = t.get_fields()

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TARGET.vocab)
    HID_DIM = 256
    ENC_LAYERS = 2
    DEC_LAYERS = 2
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(INPUT_DIM,
                  HID_DIM,
                  ENC_LAYERS,
                  ENC_HEADS,
                  ENC_PF_DIM,
                  ENC_DROPOUT)

    dec = Decoder(OUTPUT_DIM,
                  HID_DIM,
                  DEC_LAYERS,
                  DEC_HEADS,
                  DEC_PF_DIM,
                  DEC_DROPOUT)
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TARGET.vocab.stoi[TARGET.pad_token]

    return Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX), SRC, TARGET, t


def load_model(SRC, TARGET):

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TARGET.vocab)
    HID_DIM = 256
    ENC_LAYERS = 2
    DEC_LAYERS = 2
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(INPUT_DIM,
                  HID_DIM,
                  ENC_LAYERS,
                  ENC_HEADS,
                  ENC_PF_DIM,
                  ENC_DROPOUT)

    dec = Decoder(OUTPUT_DIM,
                  HID_DIM,
                  DEC_LAYERS,
                  DEC_HEADS,
                  DEC_PF_DIM,
                  DEC_DROPOUT)
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TARGET.vocab.stoi[TARGET.pad_token]

    return Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX)
