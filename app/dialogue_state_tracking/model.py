import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class FixedEmbedding(nn.Embedding):
    def __init__(self, *args, dropout=0.5, **kwargs,):
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, *args, **kwargs):
        out = super().forward(*args, *kwargs)
        return self.dropout(out)


class SelfAttention(nn.Module):
    def __init__(self, in_features, dropout=0.25):
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, H, sequence_lengths):
        """
        H = global-local encoding (which means the result of the mixture function of the local H and the global H parameterized through beta)
        lengths = the sequence length of each element of H
        In the self attention module for each i-th element of H a scalar a_i is computed which is then normalized across all a's
        """
        c = self.fc(H.contiguous().view(-1, H.size(2))
                    ).view(H.size(0), H.size(1))
        c = F.softmax(c,dim=0)
        c = c.unsqueeze(2).expand(H.size()).mul(H).sum(dim=1)
        return c


def run_rnn(rnn, inputs, lens):
    # sort by lens
    order = torch.argsort(lens).flip(0).tolist()
    reindexed = inputs.index_select(0, inputs.data.new(order).long())
    reindexed_lens = [lens[i] for i in order]
    packed = nn.utils.rnn.pack_padded_sequence(
        reindexed, reindexed_lens, batch_first=True)
    outputs, _ = rnn(packed)
    padded, _ = nn.utils.rnn.pad_packed_sequence(
        outputs, batch_first=True, padding_value=1)
    reverse_order = np.argsort(order).tolist()
    recovered = padded.index_select(0, inputs.data.new(reverse_order).long())

    return recovered


class GLADEncoder(nn.Module):
    def __init__(self, vocab_size, slots, hidden_dim=64, emb_dim=300, vectors=None, pad_idx=0, local_dropout=0.25, global_dropout=0.35):
        super().__init__()

        self.global_lstm = nn.LSTM(
            emb_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.global_attention = SelfAttention(2*hidden_dim)
        self.slots = slots
        for s in self.slots:
            setattr(self, "{}_lstm".format(s), nn.LSTM(emb_dim, hidden_dim,
                                                       bidirectional=True, batch_first=True, dropout=local_dropout))
            setattr(self, "{}_attention".format(s),
                    SelfAttention(2 * hidden_dim))
        self.beta = nn.Parameter(torch.Tensor(
            len(self.slots)), requires_grad=True)
        nn.init.normal_(self.beta)

    def beta_for_slot(self, slot_name):
        return torch.sigmoid(self.beta[self.slots.index(slot_name)])

    def forward(self, X, sequence_lengths, slot_name):

        local_lstm = getattr(self, "{}_lstm".format(slot_name))
        local_attention = getattr(self, "{}_attention".format(slot_name))
        beta = self.beta_for_slot(slot_name)

        local_h = run_rnn(local_lstm, X, sequence_lengths)
        global_h = run_rnn(self.global_lstm, X, sequence_lengths)

        h = F.dropout(local_h, 0.2, self.training)*beta + \
            (1-beta)*F.dropout(global_h, 0.2, self.training)

        c = F.dropout(local_attention(h, sequence_lengths), 0.2, self.training)*beta+(
            1-beta)*F.dropout(self.global_attention(h, sequence_lengths), 0.2, self.training)

        return h, c


def prepare_system_act_batches(system_acts, system_acts_lengths, system_acts_sequence_lengths):
    result = []
    """
    is for example 3x3x10, with batch_size of 3 with 3 sequences of 10 words each
    """
    for i in range(system_acts.size(0)):
        """
        Access to one batch, which is then 3x10, so 3 sequences with 10 words each
        """
        # 3x10 each because one
        system_act = system_acts[i]
        system_act_length = system_acts_lengths[i]
        system_acts_sequence_length = system_acts_sequence_lengths[i]

        """
        First select all relevant sequences that are in this batch based on system_act_length
        """

        relevant_sequences = system_act[:system_act_length]
        reduced_system_acts_lengths = system_acts_sequence_length[system_acts_sequence_length.nonzero(
        )].squeeze(1)
        result.append((relevant_sequences, reduced_system_acts_lengths))
    return result


class DialogueStateTrackingModel(nn.Module):

    def __init__(self, transcript_field, system_act_field, label_field, ontology, embedding_dim=300, embedding_dropout=0.2, hidden_dimension=128):
        super().__init__()
        self.ontology = ontology
        self.system_act_field = system_act_field
        self.label_field = label_field

        self.transcript_embedding = FixedEmbedding(
            len(transcript_field.vocab), embedding_dim, dropout=embedding_dropout)
        self.system_act_embedding = FixedEmbedding(
            len(system_act_field.vocab), embedding_dim, dropout=embedding_dropout)
        self.label_embedding = FixedEmbedding(
            len(label_field.vocab), embedding_dim, dropout=embedding_dropout)

        self.transcript_encoder = GLADEncoder(
            len(transcript_field.vocab), self.ontology.slots, hidden_dim=hidden_dimension)
        self.act_encoder = GLADEncoder(
            len(system_act_field.vocab), self.ontology.slots, hidden_dim=hidden_dimension)
        self.ontology_encoder = GLADEncoder(
            len(system_act_field.vocab), self.ontology.slots, hidden_dim=hidden_dimension)

        self.fc = nn.Linear(2*hidden_dimension, 1)
        self.score_weight = nn.Parameter(torch.tensor([0.5]))

    def forward(self, transcript, transcript_length, system_acts, system_acts_number, system_acts_lengths, batch_size):
        #transcript, transcript_length=batch.transcript
        #system_acts, system_acts_number, system_acts_lengths= batch.system_acts

        transcript_embeddings = self.transcript_embedding(transcript)
        num = {}
        for s, vs in self.ontology.values.items():
            num[s] = self.label_field.process([[[v] for v in vs]])

        # print(self.system_act_embedding(slot_value[0]).squeeze(0).size()) # this is correct
        ontology_embeddings = {slot_name: self.label_embedding(
            slot_values[0].squeeze(0)) for slot_name, slot_values in num.items()}
        reduced_system_acts = prepare_system_act_batches(
            system_acts, system_acts_number, system_acts_lengths)

        ys = {}
        for s in self.ontology.slots:
            # for each slot the scores need to be computed for each value

            H, c = self.transcript_encoder(
                transcript_embeddings, transcript_length, s)
            # view system acts as number_of_system_acts, seq_length, emb, dim=3x50x100 and system_acts_lengths=[4,5,6]
            _, C_acts = list(zip(*[self.act_encoder(self.system_act_embedding(system_act),
                                                    system_act_length, s) for system_act, system_act_length in reduced_system_acts]))
            _, C_ontologies = self.ontology_encoder(
                ontology_embeddings[s], num[s][2].squeeze(0), s)

            # need to zip system acts and their lengths
            y_utts = []
            q_utts = []
            for c_val in C_ontologies:
                c_val = c_val.unsqueeze(0).expand(batch_size, *c_val.size())
                scores = c_val.unsqueeze(1).expand_as(H).mul(H).sum(dim=2)
                scores = F.softmax(scores, dim=1)
                context = scores.unsqueeze(2).expand_as(H).mul(H).sum(dim=1)
                q_utts.append(context)
            y_utts = self.fc(torch.stack(q_utts, dim=1)).squeeze(2)

            q_acts = []
            for idx, c_val in enumerate(C_acts):
                c_utt = c[idx].unsqueeze(0).unsqueeze(1)
                c_val = c_val.unsqueeze(0)
                scores = c_utt.expand_as(c_val).mul(c_val).sum(dim=2)
                scores = F.softmax(scores, dim=1)
                context = scores.unsqueeze(2).expand_as(
                    c_val).mul(c_val).sum(dim=1)
                q_acts.append(context)

            y_acts = torch.cat(q_acts, dim=0).mm(C_ontologies.transpose(0, 1))
            ys[s] = torch.sigmoid(y_utts+self.score_weight*y_acts)
        return ys
