import torch
import torch.nn.functional as F


def train(model, optimizer, train_iterator):
    epoch_loss = 0
    # TODO extract scores for evaluation
    for idx, batch in enumerate(train_iterator):
        model.zero_grad()

        transcript, transcript_length = batch.transcript
        system_acts, system_acts_number, system_acts_lengths = batch.system_acts
        predictions = model(transcript, transcript_length, system_acts,
                            system_acts_number, system_acts_lengths, len(batch))

        labels = {s: [len(model.ontology.values[s])*[0]
                      for i in range(len(batch))] for s in model.ontology.slots}
        """check for each batch which labels actually occur and set them to one"""
        for idx, turn_label in enumerate(batch.turn_labels):
            for s, v in turn_label:
                labels[s][idx][model.ontology.values[s].index(v)] = 1
        labels = {s: torch.Tensor(m) for s, m in labels.items()}
        loss = 0
        for s in model.ontology.slots:
            loss += F.binary_cross_entropy(predictions[s], labels[s])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss
