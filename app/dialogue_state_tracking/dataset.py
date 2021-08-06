import torchtext.data as data
import json
from collections import defaultdict
import torch


class Ontology(object):
    def __init__(self, slots=None, values=None):
        self.slots = slots or []
        self.values = values or {}


class DialogStateTrackingDataset(data.Dataset):
    def __init__(self, dialogs_path, max_length=10, tokenize=lambda x: x.split(), include_lengths=True, batch_first=True, sos_token="<sos>", eos_token="<eos>"):
        with open(dialogs_path, "r", encoding="utf-8") as f:
            dialogs = json.loads(f.read())
        self.samples = []

        self.SOURCE = data.Field(lower=True, init_token=sos_token, eos_token=eos_token, tokenize=tokenize,
                                 batch_first=batch_first, fix_length=max_length, include_lengths=include_lengths)
        self.SYSTEM_ACT = data.Field(lower=True, init_token=sos_token, eos_token=eos_token,
                                     tokenize=tokenize, fix_length=max_length, batch_first=batch_first, dtype=torch.long)
        self.TARGET = data.NestedField(
            self.SYSTEM_ACT, include_lengths=include_lengths)
        self.LABEL = data.RawField()

        self.LABEL_VOCAB = data.NestedField(
            data.Field(), include_lengths=include_lengths)

        fields = [("transcript", self.SOURCE), ("system_transcript", self.SOURCE), ("system_acts",
                                                                                    self.TARGET), ("turn_labels", self.LABEL), ("label_vocab", self.LABEL_VOCAB)]
        slots = set()
        values = defaultdict(set)
        labels_vocab = []
        for idx, dialog in enumerate(dialogs):
            for turn in dialog["dialogue"]:
                transcript = turn["transcript"]
                system_transcript = turn["system_transcript"]
                turn_labels = []

                for turn_label in turn["turn_label"]:
                    slot, value = turn_label
                    slots.add(slot)
                    values[slot].add(value)
                    turn_labels.append([slot, value])
                    labels_vocab.append(" ".join([slot, value]))

                system_acts = []
                for system_act in turn["system_acts"]:
                    if isinstance(system_act, list):
                        slot, value = system_act
                        system_acts.append(" ".join(["inform", slot, value]))
                    else:
                        system_acts.append(" ".join(["request", system_act]))
                system_acts.append(" ".join(["<sentinel>"]))

                self.samples.append(data.Example.fromlist(
                    [transcript, system_transcript, system_acts, turn_labels, labels_vocab], fields))

        super().__init__(self.samples, fields)
        self.SOURCE.build_vocab(self)
        self.SYSTEM_ACT.build_vocab(self)
        self.TARGET.build_vocab(self)
        self.LABEL_VOCAB.build_vocab(self)
        self.ontology = Ontology(list(slots), {slot: list(
            value) for slot, value in values.items()})

    def get_fields(self):
        return self.SOURCE, self.TARGET, self.LABEL_VOCAB

    def get_ontology(self):
        return self.ontology
