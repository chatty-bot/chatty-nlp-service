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


class Seq2SeqDataset(data.Dataset):
    def __init__(self, dialogs_path, max_length=10, tokenize=lambda x: x.split(), include_lengths=True, batch_first=True, sos_token="<sos>", eos_token="<eos>"):
        with open(dialogs_path, "r", encoding="utf-8") as f:
            dialogs = json.loads(f.read())
        self.samples = []

        self.SOURCE = data.Field(lower=True, init_token=sos_token,
                                 eos_token=eos_token, tokenize=tokenize,  batch_first=batch_first)
        self.TARGET = data.Field(lower=True, init_token=sos_token,
                                 eos_token=eos_token, tokenize=tokenize,  batch_first=batch_first)

        fields = [("source", self.SOURCE), ("target", self.TARGET)]

        for idx, dialog in enumerate(dialogs):

            number_turns = len(dialog["dialogue"])
            for i in range(number_turns-1):
                current_turn = dialog["dialogue"][i]
                next_turn = dialog["dialogue"][i+1]
                """
                1. zip turn labels at position i with system output at position i+1
                """
                t = sorted(current_turn["belief_state"])
                target = next_turn["system_transcript"]
                source = " ".join([item for sublist in t for item in sublist])
                #print("Target= ",target)

                if source:
                    self.samples.append(
                        data.Example.fromlist([source, target], fields))

        super().__init__(self.samples, fields)
        self.SOURCE.build_vocab(self)
        self.TARGET.build_vocab(self)

    def get_fields(self):
        return self.SOURCE, self.TARGET
