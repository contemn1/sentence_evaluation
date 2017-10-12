from random import randint
from models import BLSTMEncoder
import numpy as np
import torch
import logging
import json
import sys
import IOUtil
import torch
from torch.utils.data import TensorDataset, DataLoader

DATA_PATH = "/home/zxj/Downloads/data"
GLOVE_PATH = DATA_PATH + "/glove.840B.300d.txt"


def load_sentences(file_path):
    try:
        with open(file_path, encoding="utf8") as file:
            sentence_dict = [json.loads(line) for line in file]
            return sentence_dict
    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)


def encoding_setences(model_path):
    model = torch.load(model_path)  #type: BLSTMEncoder

    model.set_glove_path(GLOVE_PATH)
    model.build_vocab_k_words(K=100000)
    file_path = DATA_PATH + "/en-ud-test-samples.txt"
    setence_dicts = load_sentences(file_path)

    sentences = [ele["sentence"] for ele in setence_dicts]
    embeddings = model.encode(sentences, bsize=128, tokenize=True, verbose=True)
    return embeddings


def output_encoding():
    model_path = DATA_PATH + "/infersent.allnli.pickle"
    setence_embeddings = encoding_setences(model_path)
    output_path = DATA_PATH + "/infer-sent-embeddings-test"
    np.save(output_path, setence_embeddings)

if __name__ == '__main__':
    path = "/Users/zxj/Google 云端硬盘/models_and_sample/all_positive_samples.npy"
    first = np.load(path)
    second = np.ones(first.shape[0])
    X_tensor = torch.FloatTensor(first)
    y_tensor = torch.LongTensor(second)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset=dataset, batch_size=128, shuffle=False, num_workers=2)
    for x, y in loader:
        print("x is  {0}".format(x))
        print("y is {0}".format(y))
