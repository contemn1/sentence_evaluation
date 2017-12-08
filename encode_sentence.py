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


def refined_file_to_list(file_path):
    result = []
    try:
        with open(file_path, encoding="utf8") as file:
            for sentence in file.readlines():
                result.append(sentence.rstrip())
            return result

    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)


def encoding_setences(model_path, sentence_list):
    model = torch.load(model_path, map_location=lambda storage, loc: storage)  #type: BLSTMEncoder
    glove_path = "/Users/zxj/Downloads/glove.840B.300d.txt"
    model.set_glove_path(glove_path)
    model.build_vocab_k_words(K=100000)

    embeddings = model.encode(sentence_list, bsize=128, tokenize=True, verbose=True)
    return embeddings


def output_encoding():
    model_path = DATA_PATH + "/infersent.allnli.pickle"
    setence_embeddings = encoding_setences(model_path)
    output_path = DATA_PATH + "/infer-sent-embeddings-test"
    np.save(output_path, setence_embeddings)

def test():
    path = "/Users/zxj/Google 云端硬盘/models_and_sample/all_positive_samples.npy"
    first = np.load(path)
    second = np.ones(first.shape[0])
    X_tensor = torch.FloatTensor(first)
    y_tensor = torch.LongTensor(second)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset=dataset, batch_size=128,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True)
    for x, y in loader:
        print("x is  {0}".format(x))
        print("y is {0}".format(y))


def calculate_pairwise_similarity(embedding_path):
    embd = np.load(embedding_path)
    embd = embd.reshape((int(embd.shape[0] / 3), 3, 4096))
    for ele in embd:
        res_mat = cosine_similarity(ele)
        res_need = [str(res_mat[0][1].item()), str(res_mat[1][2].item()), str(res_mat[0][2].item())]
        print("\t".join(res_need))


def sentences_unfold(file_path):
    sent_list = refined_file_to_list(file_path)
    sent_list = [ele.split("\002") for ele in sent_list]
    sent_list = [arr for arr in sent_list if len(arr) == 3]
    sent_list = [ele for arr in sent_list for ele in arr]
    return sent_list


if __name__ == '__main__':
    calculate_pairwise_similarity("/Users/zxj/PycharmProjects/sentence_evaluation/factual_embeddings.npy")
