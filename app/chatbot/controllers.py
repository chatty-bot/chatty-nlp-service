
import datetime
import torch
import pickle
import pandas as pd
import numpy as np
from app.nlp.tokenizer import get_tokenizer
from app.seq2seq.model import init_model, load_model
from app.dialogue_state_tracking.model import DialogueStateTrackingModel
from app.answer_selection.modules.qa_cnn import AnswerSelectionModule

from app.text_classification.modules.cnn import TextClassificationModel
from werkzeug.utils import secure_filename

from flask import Blueprint, request, jsonify, flash
from app import db

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext.data as data
from app.database.models import DialogueEntity, TextClassificationDataSampleEntity, IntentEntity, TrainingEntity, AnswerSelectionEntity, AnswerSelectionDataSampleEntity
import logging
import sys
import os

import collections
import time
import math


mod_chatbot = Blueprint(
    'chatbot', __name__, url_prefix='/chatbot')
prefix = "chatbot"


@mod_chatbot.route("", methods=["POST"])
def processRequest():
    chatbot_request = request.get_json()["chatbotRequest"]

    txt_clf_id = chatbot_request["txtClfId"]
    dst_id = chatbot_request["dstId"]
    answer_selection_id = chatbot_request["answSelId"]
    seq2seq_id = chatbot_request["seq2SeqId"]
    belief_state = chatbot_request.get("beliefState", {})
    X = chatbot_request["message"]
    S = chatbot_request.get("systemActs", [])

    txt_clf_module, TXT_CLF_SOURCE, TXT_CLF_TARGET = load_txt_clf_module(
        txt_clf_id)

    txt_clf_module.eval()
    txt_clf_input = TXT_CLF_SOURCE.process(
        [X.split()])
    score = txt_clf_module(txt_clf_input[0])
    predicted = torch.max(score, 1)[1]
    intent = TXT_CLF_TARGET.vocab.itos[predicted.item()]

    # first thing to do is loading the text clf module
    dst_module, DST_SOURCE, DST_TARGET, DST_LABEL, ontology = load_dst_module(
        dst_id)
    dst_input = DST_SOURCE.process(
        [X.split()])

    # s,k,l = target.process([[['request', 'confirmation'], ['<sentinel>']]])
    if not S:
        s, k, l = DST_TARGET.process([[["<sentinel>"]]])
    else:

        s, k, l = DST_TARGET.process([[S]])
    transcript, transcript_length = dst_input
    dst_module.eval()
    ys = dst_module(transcript, transcript_length, s, k, l, 1)

    prediction_belief_state = {}
    for item in ontology.values.items():
        slot_scores = [[f'{el:.2f}']
                       for el in ys[item[0]].squeeze(0).data.tolist()]
        slot_value_names = [[el] for el in item[1]]
        prediction_belief_state[item[0]] = {s[0]: v[0]
                                            for s, v in zip(slot_value_names, slot_scores)}

    actual_belief_state = _merge_belief_state(
        belief_state, prediction_belief_state)
    _generated_belief_state = " ".join(
        extract_belief(ontology, actual_belief_state))

    answ_sel_module, ANS_SEL_TEXT = load_answ_sel_module(answer_selection_id)
    answ_sel_module.eval()
    if "GREETING" == intent and not _generated_belief_state:
        # load answer selection module
        # and return a greeting
        answers = pd.read_sql_query(db.session.query(AnswerSelectionDataSampleEntity).filter_by(
            answer_selection_id=answer_selection_id, intent_name=intent).statement, db.session.bind)

        source = [ANS_SEL_TEXT.preprocess(text) for text in [X]]
        source = ANS_SEL_TEXT.pad(source)
        source = ANS_SEL_TEXT.numericalize(source)[0]

        all_answers = answers["response"].values.tolist()
        target = [ANS_SEL_TEXT.preprocess(text)
                  for text in all_answers]
        target = ANS_SEL_TEXT.pad(target)
        target = ANS_SEL_TEXT.numericalize(target)[0]

        similarity = answ_sel_module(source, target)
        values, indices = similarity.max(0)
        return jsonify({"response": all_answers[indices], "beliefState": belief_state, "systemActs": S})
    seq2seq_model, SEQ2SEQ_SOURCE, SEQ2SEQ_TARGET = load_seq2seq_module(
        seq2seq_id)

    seq2seq_model.eval()
    generated_response, attention = generate_response(
        _generated_belief_state, SEQ2SEQ_SOURCE, SEQ2SEQ_TARGET, seq2seq_model)

    response = " ".join(generated_response)
    extracted_system_act = []
    # TODO: use generated belief state to generate the system act but actually use a handcrafted rule to detect if the system wants
    # to call a doctor
    if 'arzt' in response:
        extracted_system_act.append("confirmation")
    if 'lange' in response:
        extracted_system_act.append("duration")
    return jsonify({"response": response, "beliefState": actual_belief_state, "systemActs": extracted_system_act})


def load_answ_sel_module(answer_selection_id):
    training_type = "QA_CNN"
    prefix = "ANSWER_SELECTION"
    text_file_path = "models/"+prefix+"_"+training_type + \
        "_TEXT_"+str(answer_selection_id)+".model"

    model_path = "models/"+prefix+"_"+training_type + \
        "_"+str(answer_selection_id)+".model"

    TEXT = torch.load(text_file_path)

    model = AnswerSelectionModule(
        len(TEXT.vocab), 300, pad_idx=TEXT.vocab.stoi[TEXT.pad_token])
    model.load_state_dict(torch.load(model_path))

    return model, TEXT


def generate_response(sentence, src_field, trg_field, model, max_len=50):

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


def load_seq2seq_module(seq2seq_id):
    prefix = "seq2seq"
    source_file_path = "models/"+prefix+"_"+str(seq2seq_id)+"_SOURCE.model"
    SOURCE = torch.load(source_file_path)

    target_file_path = "models/"+prefix+"_"+str(seq2seq_id)+"_TARGET.model"
    TARGET = torch.load(target_file_path)

    fileName = "models/"+prefix+"_"+str(seq2seq_id)+".model"
    model = load_model(SOURCE, TARGET)
    model.load_state_dict(torch.load(fileName))

    return model, SOURCE, TARGET


def _merge_belief_state(current, prediction, alpha=0.20):
    """
    Updating the state works as follows:
    1. If the slot has a high belief for some value that is different from the history, then it will replace the history value
        this means that the history value will become zero and the new value will keep its value
    2. If the slot is no mentioned in the new state it will just decay regulated by alpha

    This means we need a way to determine when a current belief is replaced versus a way to just decay a slot belief
    """
    if not current:
        return prediction

    for slot_name in prediction.keys():
        predicted_slot_values = prediction[slot_name]
        slot_values = current[slot_name]

        for slot_value_name in slot_values.keys():
            slot_values[slot_value_name] = round(np.clip(alpha*float(
                slot_values[slot_value_name]) + (1-alpha)*float(predicted_slot_values[slot_value_name]), 0, 1), 2)

    return current


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
    escalation_idx = np.argmax([item[1]
                                for item in state["escalation"].items()])
    duration_idx = np.argmax([item[1] for item in state["duration"].items()])
    confirmation_idx = np.argmax([item[1]
                                  for item in state["confirmation"].items()])

    # then it is neccessary to map the indices back to words

    request = is_plausible(state["request"].items(
    ), ontology, "request", request_idx, threshold)
    frequency = is_plausible(
        state["frequency"].items(), ontology, "frequency", frequency_idx, threshold)
    illness_type = is_plausible(
        state["type"].items(), ontology, "type", illness_type_idx, threshold)
    symptom = is_plausible(state["symptom"].items(
    ), ontology, "symptom", symptom_idx, threshold)

    escalation = is_plausible(state["escalation"].items(
    ), ontology, "escalation", escalation_idx, threshold)

    duration = is_plausible(state["duration"].items(
    ), ontology, "duration", duration_idx, threshold)
    confirmation = is_plausible(state["confirmation"].items(
    ), ontology, "confirmation", confirmation_idx, threshold)

    if request is not None:
        request = "request "+request
    if frequency is not None:
        frequency = "frequency "+frequency
    if illness_type is not None:
        illness_type = "type "+illness_type
    if symptom is not None:
        symptom = "symptom "+symptom
    if escalation is not None:
        escalation = "escalation "+escalation
    if duration is not None:
        duration = "duration "+duration
    if confirmation is not None:
        confirmation = "confirmation "+confirmation
    values = [request, frequency, illness_type,
              symptom, escalation, duration, confirmation]
    values = [val for val in values if val is not None]
    return sorted(values)


def load_txt_clf_module(txt_clf_id):
    prefix = "TEXT_CLASSIFICATION"
    model_type = "CNN"
    text_file_path = "models/"+prefix+"_"+model_type + \
        "_TEXT_"+str(txt_clf_id)+".model"
    label_file_path = "models/"+prefix+"_"+model_type + \
        "_LABEL_"+str(txt_clf_id)+".model"
    model_path = "models/"+prefix+"_"+model_type + \
        "_"+str(txt_clf_id)+".model"
    TEXT = torch.load(text_file_path)
    LABEL = torch.load(label_file_path)

    all_intents = db.session.query(IntentEntity).filter_by(
        classifier_id=txt_clf_id).all()

    model = TextClassificationModel(
        len(TEXT.vocab), num_classes=len(all_intents))
    model.load_state_dict(torch.load(model_path))
    return model, TEXT, LABEL


def load_dst_module(dst_id):
    prefix = "dst"
    ontology_path = "models/ontology_"+str(dst_id)+".pkl"

    with open(ontology_path, "rb") as file:
        ontology = pickle.load(file)

    source_file_path = "models/"+prefix+"_"+str(dst_id)+"_SOURCE.model"
    SOURCE = torch.load(source_file_path)

    target_file_path = "models/"+prefix+"_"+str(dst_id)+"_TARGET.model"
    TARGET = torch.load(target_file_path)

    label_file_path = "models/"+prefix+"_"+str(dst_id)+"_LABEL.model"
    LABEL = torch.load(label_file_path)

    fileName = "models/"+prefix+"_"+str(dst_id)+".model"
    model = DialogueStateTrackingModel(SOURCE, TARGET, LABEL, ontology)
    model.load_state_dict(torch.load(fileName))
    return model, SOURCE, TARGET, LABEL, ontology
