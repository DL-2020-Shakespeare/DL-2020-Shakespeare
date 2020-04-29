import glob
import os
import sys
import zipfile
from multiprocessing import cpu_count, Pool
import random as rn

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

seed = 42
rn.seed(seed)


def extract_data(extraction_dir="train", data_dir="data",
                 data_zip_name="reuters-training-corpus.zip"):
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


def get_doc(raw_xml):
    bs = BeautifulSoup(raw_xml, "lxml")
    return bs.find("headline").get_text(" ", strip=True) + " " + \
           bs.find("text").get_text(" ", strip=True)


def get_labels(raw_xml):
    labels = np.zeros(len(CODEMAP), dtype=int)
    bs = BeautifulSoup(raw_xml, "lxml")
    topics = bs.find("codes", class_="bip:topics:1.0")
    if topics:
        codes = topics.find_all("code")
        for code in codes:
            labels[CODEMAP[code["code"]]] = 1
    return labels


def worker(zip_files):
    d = {"doc": [], "labels": []}
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, "r") as zf:
            xml_files = [f for f in zf.namelist() if os.path.splitext(f)[1] == ".xml"]
            for xml_file in xml_files:
                with zf.open(xml_file, "r") as xf:
                    raw_xml = xf.read()
                    d["doc"].append(get_doc(raw_xml))
                    d["labels"].append(get_labels(raw_xml))
    return pd.DataFrame(d)


def get_docs_labels(corpus_dir, n_zip_files=None):
    pattern = os.path.join(corpus_dir, "*.zip")
    zip_files = sorted(glob.glob(pattern))
    if n_zip_files is not None:
        rn.shuffle(zip_files)
        zip_files = zip_files[:n_zip_files]
    n_cores = cpu_count()
    split = np.array_split(zip_files, n_cores)
    with Pool(n_cores) as pool:
        df_split = pool.map(worker, split)
    return pd.concat(df_split)


if __name__ == "__main__":
    df = get_docs_labels("train/REUTERS_CORPUS_2", n_zip_files=2)
    print(df.shape)
    print(df)
