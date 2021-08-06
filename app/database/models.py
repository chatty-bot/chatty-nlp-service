from app import db
db.reflect()


class TrainingEntity(db.Model):
    __tablename__ = 't_trainings'


class IntentEntity(db.Model):
    __tablename__ = 't_intents'


class TextClassificationDataSampleEntity(db.Model):
    __tablename__ = 't_txt_clf_data_samples'


class AnswerSelectionDataSampleEntity(db.Model):
    __tablename__ = 't_ans_sel_data_samples'


class AnswerSelectionEntity(db.Model):
    __tablename__ = 't_answer_selections'


class DialogueEntity(db.Model):
    __tablename__ = 't_dialogs'


class SymptomEntity(db.Model):
    __tablename__ = 't_symptoms'


class IllnessEntity(db.Model):
    __tablename__ = 't_illness'


class Seq2SeqEntity(db.Model):
    __tablename__ = 't_seq2seq'

