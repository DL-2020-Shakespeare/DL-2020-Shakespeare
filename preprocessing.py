import warnings

import en_core_web_sm
from tqdm import tqdm

warnings.filterwarnings("ignore")

nlp = en_core_web_sm.load()


def preprocess_corpus(corpus):
    preprocessed_corpus = []
    for text in tqdm(corpus):
        doc = nlp(text)
        preprocessed_text = []
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            if token.like_email:
                preprocessed_text.append("-email-")
            elif token.like_num:
                preprocessed_text.append("-num-")
            elif token.like_url:
                preprocessed_text.append("-url-")
            else:
                preprocessed_text.append(token.lemma_.lower())
        preprocessed_corpus.append(" ".join(preprocessed_text))
    return preprocessed_corpus


if __name__ == '__main__':
    for s in preprocess_corpus([
        "It's about our court-martial.",
        "They've to go.",
        "Mt. Everest's Popularity Is Still Climbing.",
        "It costs only 2.5€.",
        "I get 1500$ a month from my part-time job.",
        "My email is aaa@helsinki.fi.",
        "See you at 12:30.",
        "deals on Bre-X's stock at 1030 EST/1530 GMT",
    ]):
        print(s)
