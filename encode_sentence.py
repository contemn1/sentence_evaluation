from random import randint
from models import BLSTMEncoder
import numpy as np
import torch
import logging
import json
import sys

GLOVE_PATH = "/Users/zxj/Downloads/glove.840B.300d.txt"


def load_sentences(file_path):
    try:
        with open(file_path, encoding="utf8") as file:
            sentence_dict = [json.loads(line) for line in file]
            return sentence_dict
    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)


def encoding_test():
    model = torch.load(
        "/Users/zxj/Downloads/sentence_evaluation/InferSent/encoder/infersent.allnli.pickle",
    map_location=lambda storage, loc: storage)  #type: BLSTMEncoder

    model.set_glove_path(GLOVE_PATH)
    model.build_vocab_k_words(K=100000)
    file_path = "/Users/zxj/Downloads/sentence_evaluation/UD_English/pure-en-ud-train.txt"
    setence_dict = load_sentences(file_path)

    sentences = [ele["sentence"] for ele in setence_dict]
    embeddings = model.encode(sentences[:100], bsize=128, tokenize=True, verbose=True)


if __name__ == '__main__':
    encoding_test()