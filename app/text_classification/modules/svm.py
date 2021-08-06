import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def train_svm(classifier_id, samples, tokenize):

    X_train, X_test, y_train, y_test = train_test_split(
        samples["text"], samples["label"], test_size=0.05, random_state=41)

    # defining parameter range
    param_grid = {'C': [5, 10, 15],
                  'gamma': [ 0.01, 0.025],
                  'kernel': ['rbf']}

    grid = GridSearchCV(SVC(), param_grid, refit=True)
    text_clf = Pipeline(
        [('tfidf', TfidfVectorizer(tokenizer=tokenize)), ('svm', grid)])
    # fitting the model for grid search
    text_clf.fit(X_train, y_train)
    grid_predictions = text_clf.predict(X_test)
    # print classification report
    return text_clf,classification_report(y_test, grid_predictions, output_dict=True)
