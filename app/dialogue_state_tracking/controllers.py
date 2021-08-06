import datetime
import torch
import pickle
import pandas as pd
import numpy as np
from app.database.models import TextClassificationDataSampleEntity, IntentEntity, TrainingEntity, SymptomEntity, IllnessEntity
from app.nlp.tokenizer import get_tokenizer
from app.dialogue_state_tracking.dataset import DialogStateTrackingDataset
from app.dialogue_state_tracking.model import DialogueStateTrackingModel
from app.dialogue_state_tracking.training import train
from app.nlp.tokenizer import get_tokenizer

from flask import Blueprint, request, jsonify
from app import db

import torch
import torch.optim as optim
import torchtext.data as data
from app.database.models import DialogueEntity
import logging
import sys
import os
import collections

mod_dst = Blueprint(
    'dialogue_state_tacking', __name__, url_prefix='/dst')
prefix = "DST"


@mod_dst.route("/upload", methods=["POST"])
def upload():

    if 'file' not in request.files:
        return jsonify(success=False)

    dialogue_file = request.files["file"]

    dir_path = "datasets/"

    dialogue_file.save(os.path.join(
        dir_path, dialogue_file.filename))
    return jsonify(success=True)


@mod_dst.route("/trainDst")
def train_model():
    """
    Trains the Dialogue State Tracking Model. Accepts as query parameter 'epochs'
    """

    dst_id = request.args.get("dstId")
    file_name = request.args.get("fileName")
    epochs = int(request.args.get("epochs", 20))

    dir_path = "datasets/"
    dialog_path = os.path.join(dir_path, file_name)

    dataset = DialogStateTrackingDataset(
        dialogs_path=dialog_path, tokenize=get_tokenizer)
    source, target, label = dataset.get_fields()
    ontology = dataset.get_ontology()
    model = DialogueStateTrackingModel(source, target, label, ontology)
    train_iterator = data.BucketIterator(
        dataset, batch_size=6, sort_key=lambda x:  len(x.transcript), sort_within_batch=True)
    optimizer = optim.Adam(model.parameters())

    model.train()
    for epoch in range(epochs):

        loss = train(model, optimizer, train_iterator)
        print(loss)
    fileName = "models/"+prefix+"_"+dst_id+".model"
    torch.save(model.state_dict(), fileName)

    source_file_path = "models/"+prefix+"_"+dst_id+"_SOURCE.model"
    torch.save(source, source_file_path)

    target_file_path = "models/"+prefix+"_"+dst_id+"_TARGET.model"
    torch.save(target, target_file_path)

    label_file_path = "models/"+prefix+"_"+dst_id+"_LABEL.model"
    torch.save(label, label_file_path)

    ontology_path = "models/ontology_"+dst_id+".pkl"
    with open(ontology_path, "wb") as file:
        pickle.dump(ontology, file)
    return "ok"


@mod_dst.route("/predictDst", methods=["POST"])
def update_state():

    data = request.get_json()

    dst_id = request.args.get("dstId")
    belief_state = data.get("beliefState", {})
    X = data["transcript"]
    S = data.get("systemActs", [])
    ontology_path = "models/ontology_"+dst_id+".pkl"

    with open(ontology_path, "rb") as file:
        ontology = pickle.load(file)
    prediction_belief_state = {}
    ys = _predict(X, S, ontology, dst_id)

    for item in ontology.values.items():
        slot_scores = [[f'{el:.2f}']
                       for el in ys[item[0]].squeeze(0).data.tolist()]
        slot_value_names = [[el] for el in item[1]]
        prediction_belief_state[item[0]] = {s[0]: v[0]
                                            for s, v in zip(slot_value_names, slot_scores)}

    return _merge_belief_state(belief_state, prediction_belief_state)
    # return prediction_belief_state


def _merge_belief_state(current, prediction, alpha=0.90):
    """
    Updating the state works as follows:
    1. If the slot has a high belief for some value that is different from the history, then it will replace the history value
        this means that the history value will become zero and the new value will keep its value
    2. If the slot is no mentioned in the new state it will just decay regulated by alpha

    This means we need a way to determine when a current belief is replaced versus a way to just decay a slot belief
    """
    if not current:
        return prediction, 200

    for slot_name in prediction.keys():
        predicted_slot_values = prediction[slot_name]
        slot_values = current[slot_name]

        for slot_value_name in slot_values.keys():
            slot_values[slot_value_name] = round(np.clip(alpha*float(
                slot_values[slot_value_name]) + float(predicted_slot_values[slot_value_name]), 0, 1), 2)

    return current, 200


def is_plausible(state, ontology, slot, idx, threshold):
    value = None

    if float(list(state)[idx][1]) > threshold:
        value = list(state)[idx][0]
    return value


def extract_belief(ontology, state, threshold=0.3):
    """
    to generate a db query it is neccessary to extract the request and the informables slots
    request: symptom, frequency
    slots: symptom, type, frequency
    """

    """
    need to check if the value at argmax is bigger than a threshold
    """

    request_idx = np.argmax([item[1] for item in state["request"].items()])
    frequency_idx = np.argmax([item[1] for item in state["frequency"].items()])
    illness_type_idx = np.argmax([item[1] for item in state["type"].items()])
    symptom_idx = np.argmax([item[1] for item in state["symptom"].items()])

    # then it is neccessary to map the indices back to words

    request = is_plausible(state["request"].items(
    ), ontology, "request", request_idx, threshold)
    frequency = is_plausible(
        state["frequency"].items(), ontology, "frequency", frequency_idx, threshold)
    illness_type = is_plausible(
        state["type"].items(), ontology, "type", illness_type_idx, threshold)
    symptom = is_plausible(state["symptom"].items(
    ), ontology, "symptom", symptom_idx, threshold)

    if request is not None:
        request = "request "+request
    if frequency is not None:
        frequency = "frequency "+frequency
    if illness_type is not None:
        illness_type = "type "+illness_type
    if symptom is not None:
        symptom = "symptom "+symptom
    values = [request, frequency, illness_type, symptom]
    values = [val for val in values if val is not None]
    return sorted(values)


def _predict(X, S, ontology, dst_id):

    source_file_path = "models/"+prefix+"_"+dst_id+"_SOURCE.model"
    SOURCE = torch.load(source_file_path)

    target_file_path = "models/"+prefix+"_"+dst_id+"_TARGET.model"
    TARGET = torch.load(target_file_path)

    label_file_path = "models/"+prefix+"_"+dst_id+"_LABEL.model"
    LABEL = torch.load(label_file_path)

    fileName = "models/"+prefix+"_"+dst_id+".model"
    model = DialogueStateTrackingModel(SOURCE, TARGET, LABEL, ontology)
    model.load_state_dict(torch.load(fileName))

    X = SOURCE.process(
        [X.split()])

    # s,k,l = target.process([[['request', 'confirmation'], ['<sentinel>']]])
    if not S:
        s, k, l = TARGET.process([[["<sentinel>"]]])
    else:
        s, k, l = TARGET.process([S])
    transcript, transcript_length = X
    model.eval()
    ys = model(transcript, transcript_length, s, k, l, 1)

    return ys
