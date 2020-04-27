import pickle
import pprint

import autokeras as ak
import numpy as np

seed = 42
np.random.seed(seed)

with open("train/preprocessed_docs.pkl", "rb") as f:
    x_train = np.array(pickle.load(f))
y_train = np.load("train/labels.npy")

print(x_train.shape)
print(y_train.shape)

clf = ak.TextClassifier(max_trials=100, multi_label=True, seed=seed, overwrite=True, directory="tmp")
clf.fit(x_train, y_train)

model = clf.export_model()
print(type(model))
model.summary(line_length=120)
pprint.PrettyPrinter(indent=2).pprint(model.get_config())
