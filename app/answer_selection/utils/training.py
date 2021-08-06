from app.answer_selection.modules.qa_cnn import load_data, train, AnswerSelectionDataset, AnswerSelectionModule
from app import db
from app.database.models import TrainingEntity, AnswerSelectionDataSampleEntity
from app.nlp.tokenizer import get_tokenizer
import datetime
import torch

training_type = "QA_CNN"
prefix = "ANSWER_SELECTION"


def train_model(answer_selection_id, epochs=5, fix_length=150):
    query = db.session.query(AnswerSelectionDataSampleEntity).filter_by(
        answer_selection_id=answer_selection_id).statement
    dataset, iterator, TEXT = load_data(
        query, get_tokenizer)

    model = AnswerSelectionModule(
        len(TEXT.vocab), 300, pad_idx=TEXT.vocab.stoi[TEXT.pad_token])

    training = TrainingEntity(
        answer_selection_id=answer_selection_id, training_type=training_type, training_status="IN_PROGRESS", started_at=datetime.datetime.now())
    db.session.add(training)
    db.session.commit()

    train(model, dataset, iterator, TEXT, epochs)

    training.training_status = "DONE"
    db.session.commit()

    # after training set training entity status to finished
    text_file_path = "models/"+prefix+"_"+training_type + \
        "_TEXT_"+str(answer_selection_id)+".model"

    torch.save(TEXT, text_file_path)

    fileName = "models/"+prefix+"_"+training_type + \
        "_"+str(answer_selection_id)+".model"
    torch.save(model.state_dict(), fileName)
