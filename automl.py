import argparse
import pickle
import pprint
import warnings

warnings.filterwarnings("ignore")

import autokeras as ak
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

seed = 42


def main(args):
    n_docs = args.n_docs

    df = pd.read_pickle("train/data.pkl")
    labels = np.array(df["labels"].tolist())
    with open("train/preprocessed_docs_no_sw_no_rep.pkl", "rb") as f:
        preprocessed_docs = pickle.load(f)

    x_train, x_test, y_train, y_test = train_test_split(preprocessed_docs,
                                                        labels,
                                                        train_size=n_docs,
                                                        test_size=n_docs,
                                                        random_state=seed)
    print(len(x_train))
    print(y_train.shape)
    print(len(x_test))
    print(y_test.shape)

    clf = ak.TextClassifier(max_trials=100,
                            multi_label=True,
                            loss="binary_crossentropy",
                            seed=seed,
                            overwrite=True,
                            directory="/scratch/project_2002961/")
    clf.fit(np.array(x_train), y_train)

    for model in clf.tuner.get_best_models(10):
	y_pred = np.round(model.predict(x_test)).astype(int)
	print(f"test avg. accuracy: {np.mean([accuracy_score(y_test[i], y_pred[i]) for i in range(len(y_test))])}")
	print(f"test hamming loss: {hamming_loss(y_test, y_pred)}")
        model.summary(line_length=120)
        pprint.PrettyPrinter(indent=2).pprint(model.get_config())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_docs", type=int)
    args = parser.parse_args()
    main(args)
