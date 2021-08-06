
def get_tokenizer(words):
    return [t for t in words.split()]
    # return [tok.text.lower() for tok in nlp.tokenizer(words)]
