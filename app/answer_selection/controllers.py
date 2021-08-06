from flask import Blueprint, request
from app.answer_selection.utils.training import train_model
from app.answer_selection.modules.qa_cnn import AnswerSelectionModule
from app.database.models import AnswerSelectionEntity, AnswerSelectionDataSampleEntity
from app.answer_selection.utils.training import prefix, training_type
from app import db
import torch
import pandas as pd
mod_answer_selection = Blueprint(
    'answer_selection', __name__, url_prefix='/answer_selection')


@mod_answer_selection.route("/train/<int:answer_selection_id>")
def train(answer_selection_id):
    # check if answer selection module exists in db

    epochs = int(request.args.get("epochs"))

    answer_selection_module_exists = db.session.query(
        AnswerSelectionEntity).filter_by(id=answer_selection_id).first()

    if answer_selection_module_exists is not None:
        train_model(answer_selection_id, epochs)
        return '', 200
    else:
        return "Answer selection Module not found", 404


@mod_answer_selection.route("/predict/<int:answer_selection_id>", methods=["POST"])
def predict(answer_selection_id):
    data = request.get_json()

    x = data["inputs"]
    relevant_intent_name = request.args.get("intent_name")
    # load all necessary models
    # select all t_ans_sel_data_samples that correspond to that answer_selection id
    text_file_path = "models/"+prefix+"_"+training_type + \
        "_TEXT_"+str(answer_selection_id)+".model"

    model_path = "models/"+prefix+"_"+training_type + \
        "_"+str(answer_selection_id)+".model"

    answers = pd.read_sql_query(db.session.query(AnswerSelectionDataSampleEntity).filter_by(
        answer_selection_id=answer_selection_id, intent_name=relevant_intent_name).statement, db.session.bind)
    TEXT = torch.load(text_file_path)

    model = AnswerSelectionModule(
        len(TEXT.vocab), 300, pad_idx=TEXT.vocab.stoi[TEXT.pad_token])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    source = [TEXT.preprocess(text) for text in x]
    source = TEXT.pad(source)
    source = TEXT.numericalize(source)[0]

    all_answers = answers["response"].values.tolist()
    target = [TEXT.preprocess(text)
              for text in all_answers]
    target = TEXT.pad(target)
    target = TEXT.numericalize(target)[0]

    similarity = model(source, target)
    values, indices = similarity.max(0)
    print(all_answers[indices])
    return all_answers[indices], 200
