import argparse
import pickle
import pprint

import autokeras as ak
import numpy as np

seed = 42
np.random.seed(seed)


def main(args):
    n_docs = args.n_docs

    with open("train/preprocessed_docs.pkl", "rb") as f:
        x_train = np.array(pickle.load(f))
    y_train = np.load("train/labels.npy")

    if n_docs is not None:
        x_train = x_train[:n_docs]
        y_train = y_train[:n_docs]

    print(x_train.shape)
    print(y_train.shape)

    clf = ak.TextClassifier(max_trials=100, multi_label=True, seed=seed, overwrite=True, directory="/scratch/project_2002961/")
    clf.fit(x_train, y_train)

    model = clf.export_model()
    print(type(model))
    model.summary(line_length=120)
    pprint.PrettyPrinter(indent=2).pprint(model.get_config())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_docs", type=int)
    args = parser.parse_args()
    main(args)
