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
from sklearn.metrics.pairwise import cosine_similarity

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


def read_file(file_path, preprocess=lambda x: x):
    try:
        with open(file_path, encoding="utf8") as file:
            for sentence in file.readlines():
                yield (preprocess(sentence))

    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)

def resume_model(model_path):
    model = torch.load(model_path, map_location=lambda storage, loc: storage)  #type: BLSTMEncoder
    glove_path = "/Users/zxj/Downloads/glove.840B.300d.txt"
    model.set_glove_path(glove_path)
    model.build_vocab_k_words(K=100000)
    return model

def encoding_setences(model_path, sentence_list):
    model = resume_model(model_path)  #type: BLSTMEncoder
    embeddings = model.encode(sentence_list, bsize=128, tokenize=True, verbose=True)
    return embeddings

def output_encoding():
    model_path = DATA_PATH + "/infersent.allnli.pickle"
    setence_embeddings = encoding_setences(model_path)
    output_path = DATA_PATH + "/infer-sent-embeddings-test"
    np.save(output_path, setence_embeddings)


def calculate_pairwise_similarity(embedding_path):
    embd = np.load(embedding_path)
    embd = embd.reshape((int(embd.shape[0] / 3), 3, 4096))
    for ele in embd:
        res_mat = cosine_similarity(ele)
        res_need = [str(res_mat[0][1].item()), str(res_mat[1][2].item()), str(res_mat[0][2].item())]
        print("\t".join(res_need))


def sentences_unfold(file_path):
    sent_list = read_file(file_path)
    sent_list = [ele.split("\002") for ele in sent_list]
    sent_list = [arr for arr in sent_list if len(arr) == 3]
    sent_list = [ele for arr in sent_list for ele in arr]
    return sent_list


if __name__ == '__main__':
    sick_path = "/Users/zxj/Downloads/SICK/SICK.txt"
    file_list = read_file(sick_path)
    file_list = (ele.split("\t")[1:7] for ele in file_list if not ele.startswith("pair_ID"))
    file_list = ([ele[0], ele[1], ele[3]] for ele in file_list if ele[2] == "ENTAILMENT")
    file_list = list(file_list)
    first = [ele[0] for ele in file_list]
    second = [ele[1] for ele in file_list]
    third = [ele[2] for ele in file_list]
    model_path = "/Users/zxj/Downloads/sentence_evaluation/InferSent/encoder/infersent.allnli.pickle"
    model = resume_model(model_path)
    first_emb = model.encode(first, bsize=128, tokenize=True, verbose=True)
    second_emb = model.encode(second, bsize=128, tokenize=True, verbose=True)
    res = cosine_similarity(first_emb, second_emb).diagonal().tolist()
    for ele in zip(res, third):
        print(str(ele[0]) + "\t" + ele[1])