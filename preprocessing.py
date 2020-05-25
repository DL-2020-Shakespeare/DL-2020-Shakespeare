import warnings
from multiprocessing import cpu_count, Pool

import en_core_web_sm
import numpy as np

warnings.filterwarnings("ignore")

nlp = en_core_web_sm.load()


def worker(corpus):
    preprocessed_corpus = []
    for text in corpus:
        doc = nlp(str(text))
        preprocessed_text = []
        for token in doc:
#             if not token.is_space:
#                 preprocessed_text.append(token.text)
            if token.is_punct or token.is_space or token.is_stop:
                continue
            else:
                preprocessed_text.append(token.lemma_.lower())
        preprocessed_corpus.append(" ".join(preprocessed_text))
    return preprocessed_corpus


def preprocess_corpus(corpus):
    n_cores = cpu_count()
    split = np.array_split(corpus, n_cores)
    with Pool(n_cores) as pool:
        split = pool.map(worker, split)
    return np.concatenate(split).tolist()


if __name__ == "__main__":
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
