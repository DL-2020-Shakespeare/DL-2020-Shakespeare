import glob
import os
import sys
import zipfile

import numpy as np
from bs4 import BeautifulSoup


def extract_data(extraction_dir="train", data_dir="data", data_zip_name="reuters-training-corpus.zip"):
    root_dir = os.getcwd()
    if root_dir not in sys.path:
        sys.path.append(root_dir)

    data_dir_path = os.path.abspath(os.path.join(root_dir, data_dir))
    data_path = os.path.abspath(os.path.join(data_dir_path, data_zip_name))

    with zipfile.ZipFile(data_path) as zip_f:
        zip_f.extractall(extraction_dir)


def get_codes(codefile):
    codes = {}
    i = 0
    with open(codefile, "r") as cf:
        for line in cf:
            if not line.startswith(";"):
                code = line.strip().split("\t")[0]
                codes[code] = i
                i += 1
    return codes


CODEMAP = get_codes("data/topic_codes.txt")


def get_text(doc):
    bs = BeautifulSoup(doc, "lxml")
    text_field = bs.find("text")
    if text_field:
        p_fields = text_field.find_all("p")
        return " ".join([p_field.contents[0] for p_field in p_fields])
    else:
        return ""


def get_labels(doc):
    vec = np.zeros(len(CODEMAP), dtype=int)
    bs = BeautifulSoup(doc, "lxml")
    topics = bs.find("codes", class_="bip:topics:1.0")
    if topics:
        codes = topics.find_all("code")
        for code in codes:
            vec[CODEMAP[code["code"]]] = 1
    return vec


def get_doc_labels(corpus_dir):
    pattern = os.path.join(corpus_dir, "*.zip")
    doc_labels = {}
    for zfile in sorted(glob.glob(pattern)):
        with zipfile.ZipFile(zfile, "r") as zf:
            for xmlfile in zf.namelist():
                with zf.open(xmlfile, "r") as xf:
                    doc_labels[xmlfile] = get_labels(xf.read())
    vecs = np.empty((len(doc_labels), len(CODEMAP)), dtype=int)
    for i, (doc, label) in enumerate(sorted(doc_labels.items())):
        vecs[i] = label
    return vecs


def get_docs_labels(corpus_dir):
    pattern = os.path.join(corpus_dir, "*.zip")
    texts_dict = {}
    labels_dict = {}
    # *** for testing ***
    # for zfile in sorted(glob.glob(pattern))[:1]:
    for zfile in sorted(glob.glob(pattern)):
        with zipfile.ZipFile(zfile, "r") as zf:
            # *** for testing ***
            # for xmlfile in zf.namelist()[:1]:
            for xmlfile in zf.namelist():
                if os.path.splitext(xmlfile)[1] == ".xml":
                    with zf.open(xmlfile, "r") as xf:
                        doc = xf.read()
                        texts_dict[xmlfile] = get_text(doc)
                        labels_dict[xmlfile] = get_labels(doc)
    texts = [text for (_, text) in sorted(texts_dict.items())]
    labels_mat = np.empty((len(labels_dict), len(CODEMAP)), dtype=int)
    for i, (_, labels) in enumerate(sorted(labels_dict.items())):
        labels_mat[i] = labels
    return texts, labels_mat
