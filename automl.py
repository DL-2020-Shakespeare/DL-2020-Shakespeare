import argparse
import pickle
import pprint
import warnings

warnings.filterwarnings("ignore")

import autokeras as ak
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, accuracy_score, \
    label_ranking_average_precision_score, ndcg_score, f1_score
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
        y_pred_prob = model.predict(x_test)
        y_pred = np.round(y_pred_prob)

        print(f"accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"AP (macro): {average_precision_score(y_test, y_pred_prob, average='macro')}")
        print(f"AP (micro): {average_precision_score(y_test, y_pred_prob, average='micro')}")
        print(f"AP (samples): {average_precision_score(y_test, y_pred_prob, average='samples')}")
        print(f"AP (weighted): {average_precision_score(y_test, y_pred_prob, average='weighted')}")
        print(f"F1 (macro): {f1_score(y_test, y_pred, average='macro')}")
        print(f"F1 (micro): {f1_score(y_test, y_pred, average='micro')}")
        print(f"F1 (samples): {f1_score(y_test, y_pred, average='samples')}")
        print(f"F1 (weighted): {f1_score(y_test, y_pred, average='weighted')}")
        print(f"LRAP: {label_ranking_average_precision_score(y_test, y_pred_prob)}")
        print(f"NDCG: {ndcg_score(y_test, y_pred_prob)}")

        model.summary(line_length=120)
        pprint.PrettyPrinter(indent=2).pprint(model.get_config())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_docs", type=int)
    args = parser.parse_args()
    main(args)
