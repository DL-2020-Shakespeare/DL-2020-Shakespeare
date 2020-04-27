import pickle

import autokeras as ak
import numpy as np

seed = 42
np.random.seed(seed)

with open("train/preprocessed_docs.pkl", "rb") as f:
    x_train = np.array(pickle.load(f))
y_train = np.load("train/labels.npy")

clf = ak.TextClassifier(max_trials=100, multi_label=True, seed=seed)
clf.fit(x_train, y_train)

print(clf.export_model().summary())
