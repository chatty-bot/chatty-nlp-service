
import datetime
import torch
import pickle
import pandas as pd
import numpy as np
from app.nlp.tokenizer import get_tokenizer
from app.seq2seq.model import init_model, load_model
from werkzeug.utils import secure_filename

from flask import Blueprint, request, jsonify, flash
from app import db

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext.data as data
from app.database.models import DialogueEntity
import logging
import sys
import os
import collections
import time
import math


mod_seq2seq = Blueprint(
    'seq2eq', __name__, url_prefix='/seq2seq')
prefix = "seq2seq"


@mod_seq2seq.route("/upload", methods=["POST"])
def upload():

    if 'file' not in request.files:
        return jsonify(success=False)

    dialogue_file = request.files["file"]

    dir_path = "datasets/"

    dialogue_file.save(os.path.join(
        dir_path, dialogue_file.filename))
    return jsonify(success=True)


@mod_seq2seq.route("/trainSeq2Seq")
def train_model():
    epochs = int(request.args.get("epochs", 20))
    file_name = request.args.get("fileName")
    seq2seq_id = request.args.get("seq2SeqId")
    model, SRC, TARGET, dataset = init_model(file_name)
    train_iterator = data.BucketIterator(dataset, batch_size=8, sort_key=lambda x:  len(
        x.source), sort_within_batch=True, shuffle=True)

    LEARNING_RATE = 0.0005

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(
        ignore_index=TARGET.vocab.stoi[TARGET.pad_token])
    N_EPOCHS = epochs
    CLIP = 1

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        # valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # if valid_loss < best_valid_loss:
        # best_valid_loss = valid_loss
        # torch.save(model.state_dict(), 'tut6-model.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    fileName = "models/"+prefix+"_"+seq2seq_id+".model"
    torch.save(model.state_dict(), fileName)

    source_file_path = "models/"+prefix+"_"+seq2seq_id+"_SOURCE.model"
    torch.save(SRC, source_file_path)

    target_file_path = "models/"+prefix+"_"+seq2seq_id+"_TARGET.model"
    torch.save(TARGET, target_file_path)
    return "", 200


@mod_seq2seq.route("/predictSeq2Seq", methods=["POST"])
def predict():
    seq2seq_id = request.args.get("seq2SeqId")
    data = request.get_json()
    X = data["transcript"]
    source_file_path = "models/"+prefix+"_"+seq2seq_id+"_SOURCE.model"
    SOURCE = torch.load(source_file_path)

    target_file_path = "models/"+prefix+"_"+seq2seq_id+"_TARGET.model"
    TARGET = torch.load(target_file_path)

    fileName = "models/"+prefix+"_"+seq2seq_id+".model"
    model = load_model(SOURCE, TARGET)
    model.load_state_dict(torch.load(fileName))

    # X = SOURCE.process(
    # [X.split()])
    model.eval()
    generated_response, attention = translate_sentence(
        X, SOURCE, TARGET, model)
    print(generated_response)
    return " ".join(generated_response)


def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src = batch.source
        trg = batch.target

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def evaluate(model, iterator, criterion):

    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.source[0]
            trg = batch.target[0]

            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]
            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def translate_sentence(sentence, src_field, trg_field, model, max_len=50):

    model.eval()
    if isinstance(sentence, str):
        tokens = [text.lower() for text in sentence.split()]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(
                trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention
