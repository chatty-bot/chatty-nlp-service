
from flask import Flask

from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://qa:qa@localhost:5432/qa'
db = SQLAlchemy(app)


from app.answer_selection.controllers import mod_answer_selection
from app.text_classification.controllers import mod_text_classification
from app.dialogue_state_tracking.controllers import mod_dst
from app.seq2seq.controllers import mod_seq2seq
from app.chatbot.controllers import mod_chatbot

# Register blueprint(s)
app.register_blueprint(mod_answer_selection)
app.register_blueprint(mod_text_classification)
app.register_blueprint(mod_dst)
app.register_blueprint(mod_seq2seq)
app.register_blueprint(mod_chatbot)

@app.route("/ping")
def ping():
    return "pong"
