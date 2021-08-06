import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchtext.data as data
import pandas as pd
from app import db
from torchtext.data import Field, LabelField, Iterator


def load_data(data, tokenize, batch_size=1, fix_length=50):
    TEXT = Field(tokenize=tokenize, lower=True,
                 fix_length=fix_length, include_lengths=True, batch_first=True)

    fields = [("text", TEXT), ("response", TEXT), ("invalid", TEXT)]
    dataset = AnswerSelectionDataset(data, fields=fields)
    TEXT.build_vocab(dataset)

    iterator = Iterator(dataset, batch_size=batch_size, shuffle=True)
    return dataset, iterator, TEXT


class AnswerSelectionDataset(data.Dataset):
    def __init__(self, query, fields):
        df = pd.read_sql_query(query, db.session.bind)
        examples = []
        data_length = len(df)
        wrong_answer_idx = torch.randint(0, data_length, (1,)).item()

        for i in range(data_length):
            examples.append(data.Example.fromlist([df["text"][i], df["response"][i], df["response"][wrong_answer_idx]],
                                                  fields))
        super().__init__(examples, fields)


class AnswerSelectionModule(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, num_features=200, num_filters=2000, filter_size=2, dropout=0.5, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(
            vocabulary_size, embedding_dim=embedding_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim, num_features)
        self.activation = nn.Tanh()
        self.conv = nn.Conv2d(1, out_channels=num_filters,
                              kernel_size=(filter_size, num_features))
        self.dropout = nn.Dropout(p=dropout)
        self.ranking = nn.CosineSimilarity()

    def forward(self, x, y):
        # BATCH_SIZE, SENTENCE_LENGTH; EMBEDDING_DIM
        x_emb = self.emb(x).unsqueeze(1)
        #x_emb = self.tanh(x_emb)
        y_emb = self.emb(y).unsqueeze(1)
        #y_emb = self.tanh(y_emb)
        x_fc = self.activation(self.fc(x_emb))

        x_conv = self.conv(x_fc).squeeze(3)

        x_max_pooled = self.activation(F.max_pool1d(x_conv, x_conv.size(2)))

        y_fc = self.activation(self.fc(y_emb))

        y_conv = self.conv(y_fc).squeeze(3)
        y_max_pooled = self.activation(F.max_pool1d(y_conv, y_conv.size(2)))

        x_y_sim = self.ranking(x_max_pooled, y_max_pooled)
        return x_y_sim


def hinge(x1, x2, margin=.009):
    loss = torch.max(torch.tensor(0, dtype=torch.float32), margin-x1+x2)
    return loss.mean()


def train(model, dataset, iterator, source_field, epochs=20):
    print("Start QA Model Training")
    margin = 0.009
    model.train()
    optimizer = optim.SGD(
        model.parameters(), lr=0.01, weight_decay=0.0001)
    criterion = hinge
    for epoch in range(epochs):

        total_epoch_loss = 0
        for idx, batch in enumerate(iterator):
            optimizer.zero_grad()

            source = batch.text[0]
            response = batch.response[0]
            invalid_response = batch.invalid[0]

            pos_sim = model(source, response)
            neg_sim = model(source, invalid_response)
            resamples = 0

            while pos_sim-neg_sim >= margin and resamples < 50:
                # sample a new negative
                wrong_answer_idx = torch.randint(
                    0, len(iterator), (1,)).item()
                invalid_response = dataset[wrong_answer_idx].response
                invalid_response = source_field.preprocess(invalid_response)
                invalid_response = source_field.pad([invalid_response])
                invalid_response = source_field.numericalize(invalid_response)[
                    0]
                neg_sim = model(source, invalid_response)
                resamples += 1

            loss = criterion(pos_sim, neg_sim)
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()

        print(total_epoch_loss)


def eval():
    pass  # TODO
