from app.text_classification.modules.svm import train_svm
from app.text_classification.modules.cnn import train_cnn
import torch
prefix = "TEXT_CLASSIFICATION"


def train_model(training_type, classifier_id, data, intents, tokenize=lambda x: x.split(), epochs=100):
    model, metrics = None, None
    batch_size = 20
    if training_type == "SVM":
        # start svm training

        model, metrics = train_svm(classifier_id, data, tokenize)

    elif training_type == "CNN":

        model, metrics, TEXT, LABEL = train_cnn(
            data, tokenize, len(intents), epochs, batch_size)
        text_file_path = "models/"+prefix+"_"+training_type + \
            "_TEXT_"+str(classifier_id)+".model"
        label_file_path = "models/"+prefix+"_"+training_type + \
            "_LABEL_"+str(classifier_id)+".model"
        torch.save(TEXT, text_file_path)
        torch.save(LABEL, label_file_path)
    else:
        pass
    return model, metrics
