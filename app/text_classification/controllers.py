import datetime
import torch
import pickle
import pandas as pd
from app.database.models import TextClassificationDataSampleEntity, IntentEntity, TrainingEntity
from app.nlp.tokenizer import get_tokenizer

from app.text_classification.modules.cnn import TextClassificationModel
from app.text_classification.utils.training import train_model
from flask import Blueprint, request, jsonify
from app import db


mod_text_classification = Blueprint(
    'text_classification', __name__, url_prefix='/text_classification')
prefix = "TEXT_CLASSIFICATION"


@mod_text_classification.route("/train/<int:classifier_id>")
def train(classifier_id):

    training_type = request.args.get("training_type")
    epochs = int(request.args.get("epochs"))

    if training_type is None:
        return "A training type is required as query parameter", 400

    all_samples = pd.read_sql_query(db.session.query(TextClassificationDataSampleEntity).filter_by(
        classifier_id=classifier_id).statement, db.session.bind)

    all_intents = db.session.query(IntentEntity).filter_by(
        classifier_id=classifier_id).all()

    training = TrainingEntity(
        classifier_id=classifier_id, training_type=training_type, training_status="IN_PROGRESS", started_at=datetime.datetime.now())
    db.session.add(training)
    db.session.commit()
    model, report = None, None

    try:
        # (training_type, classifier_id, data, tokenize=lambda x: x.split())
        model, report = train_model(
            training_type, classifier_id, all_samples, all_intents, get_tokenizer, epochs)
        training.training_status = "DONE"
        db.session.commit()
    except ValueError:

        training.training_status = "FAILED"
        db.session.commit()

    if model is not None and report is not None:
        if training_type == "CNN":
            fileName = "models/"+prefix+"_"+training_type + \
                "_"+str(classifier_id)+".model"
            torch.save(model.state_dict(), fileName)
            return jsonify(report)
        elif training_type == "SVM":
            fileName = "models/"+training_type+"_"+str(classifier_id)+".pkl"
            with open(fileName, "wb") as file:
                pickle.dump(model, file)

            result_list = []
            for intent in all_intents:
                intent_name = intent.intent_name
                if report[intent_name] is not None:
                    score = report[intent_name]
                    score["intent_name"] = intent_name
                    result_list.append(
                        {"intent_name": intent_name, "precision": score["precision"]})

            return jsonify(result_list)
    else:
        training.in_progress = False
        training.training_status = "FAILED"
        db.session.commit()
        return "Training failed", 400

    # model_cnn, report_cnn = train_cnn(
        # classifier_id, all_samples, all_intents, tokenize)


@mod_text_classification.route("/predict/<int:classifier_id>", methods=["POST"])
def predict(classifier_id):

    data = request.get_json()

    model_types = data["type"]
    x = data["inputs"]
    result = []

    all_intents = db.session.query(IntentEntity).filter_by(
        classifier_id=classifier_id).all()

    for model_type in model_types:
        print(model_type)
        if model_type == "CNN":

            text_file_path = "models/"+prefix+"_"+model_type + \
                "_TEXT_"+str(classifier_id)+".model"
            label_file_path = "models/"+prefix+"_"+model_type + \
                "_LABEL_"+str(classifier_id)+".model"
            model_path = "models/"+prefix+"_"+model_type + \
                "_"+str(classifier_id)+".model"
            TEXT = torch.load(text_file_path)
            LABEL = torch.load(label_file_path)

            model = TextClassificationModel(
                len(TEXT.vocab), num_classes=len(all_intents))
            model.load_state_dict(torch.load(model_path))

            model.eval()
            tokens = [TEXT.preprocess(text) for text in x]
            padded = TEXT.pad(tokens)
            numericalize = TEXT.numericalize(padded)
            score = model(numericalize[0])
            predicted = torch.max(score, 1)[1]
            result.append({"model_type": model_type, "predictions": [
                          LABEL.vocab.itos[p] for p in predicted]})

        if model_type == "SVM":
            with open("models/"+model_type+"_"+str(classifier_id)+".pkl", "rb") as file:
                model = pickle.load(file)
            predictions = model.predict(x)
            result.append(
                {"model_type": model_type, "predictions": predictions.tolist()})

    return jsonify(result)
