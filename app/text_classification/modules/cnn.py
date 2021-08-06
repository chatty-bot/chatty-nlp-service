
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext.data import Field, LabelField, Iterator
from app.text_classification.utils.datasets import DataFrameDataset


def load_data(data, tokenize, batch_size=16, fix_length=50):
    TEXT = Field(tokenize=tokenize, lower=True,
                 fix_length=fix_length, include_lengths=True, batch_first=True)
    LABEL = LabelField()
    fields = [("text", TEXT), ("label", LABEL)]
    train, test = DataFrameDataset(data, fields=fields).split()
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    train_iterator = Iterator(train, batch_size=batch_size, shuffle=True)
    test_iterator = Iterator(test, batch_size=batch_size, shuffle=True)
    return train_iterator, test_iterator, TEXT, LABEL


class TextClassificationModel(nn.Module):
    def __init__(self, vocabulary_size, num_classes, embedding_dim=300, weights=None, out_channels=100, filter_sizes=[2, 3, 4]):
        super(TextClassificationModel, self).__init__()
        self.VOCAB_SIZE = vocabulary_size
        self.EMB_DIM = embedding_dim

        self.emb = nn.Embedding(self.VOCAB_SIZE, self.EMB_DIM)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(
            size, self.EMB_DIM)) for size in filter_sizes])
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(len(filter_sizes)*out_channels, num_classes)

        if weights is not None:
            self.emb = nn.Embedding.from_pretrained(weights)

    def forward(self, input):
        # input = BATCH_SIZE,FIXED_LENGTH
        input_embedding = self.emb(input)
        # input_embedding = BATCH_SIZE,FIXED_LENGTH, EMB_DIM TODO verify dimensions
        # unsqueeze input-embedding before passing tp conv layers
        input_embedding = input_embedding.unsqueeze(1)
        # input_embedding = BATCH_SIZE, 1, FIXED_LENGTH, EMB_DIM

        # After applying convolution size is now
        # BATCH_SIZE, OUT_CHANNELS, FIXED_LENGTH, 1 --> so dimension at index 3 can be removed by calling squeeze
        input_convs = [F.relu(conv(input_embedding)).squeeze(3)
                       for conv in self.convs]
        # item in input_convs = BATCH_SIZE, OUT_CHANNELS, FIXED_LENGTH, 1 , because kernel size reduces EMB_DIM to 1

        # Next max pooling is applied
        # Remember we have a sample with dimensions : BATCH_SIZE, OUT_CHANNELS, FIXED_LENGTH
        # so max pooling will collapse (OUT_CHANNEL, FIXED_LENGTH) --> (OUT_CHANNEL, 1) (keeping the maximum value)
        input_max_pool = [F.max_pool1d(
            x, x.size(2)).squeeze(2) for x in input_convs]

        # concatenate each of the layers so we have: BATCH_SIZE, LEN(filter_sizes)*OUT_CHANNELS
        input_max_pool = torch.cat(input_max_pool, dim=1)

        # Then apply dropout, which means that half the neurons will output 0
        input_dropout = self.dropout(input_max_pool)

        # At the end use a linear (or dense) layer.
        # Input in this layer is: BATCH_SIZE,LEN(filter_sizes)*OUT_CHANNELS
        # and this returns BATCH_SIZE,NUM_CLASSES
        return self.linear(input_dropout)


def train(model, train_iterator, epochs):
    for epoch in range(epochs):
        model.train()

        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        total_epoch_loss = 0
        for idx, batch in enumerate(train_iterator):
            optimizer.zero_grad()  # TODO why is this called?

            text = batch.text[0]  # because we set included_length= True
            label = batch.label

            prediction = model(text)

            loss = criterion(prediction, label)

            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()
        print(total_epoch_loss)


def eval(model, test_iterator, num_classes):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        total_epoch_accuracy = 0
        total_epoch_loss = 0
        for idx, batch in enumerate(test_iterator):
            text = batch.text[0]
            label = batch.label

            prediction = model(text)
            loss = criterion(prediction, label)

            predicted_classes = prediction.argmax(1)
            correct += (prediction.argmax(1) == label).sum().item()

            for c, p in zip(label, predicted_classes):
                confusion_matrix[c, p] += 1

    return confusion_matrix.diag()/confusion_matrix.sum(1)


def train_cnn(data, tokenize, num_classes, epochs, batch_size):
    data_iterator, test_iterator, TEXT, LABEL = load_data(
        data, tokenize, batch_size=batch_size)
    model = TextClassificationModel(
        len(TEXT.vocab), num_classes=num_classes)

    train(model, data_iterator, epochs)
    metrics = eval(model, test_iterator, num_classes)
    scores = []
    for idx, val in enumerate(metrics):
        scores.append(
            {"intent_name": LABEL.vocab.itos[idx], "precision": val.item()})

    return model, scores, TEXT, LABEL
