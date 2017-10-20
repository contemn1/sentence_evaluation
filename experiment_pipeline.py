import numpy as np
import json
from binary import BinaryClassifierEval
import torch
from torch.utils.data import DataLoader
from customized_dataset import TextDataset
from word_dict_test import read_file
from IOUtil import unfold_domain
import IOUtil
import random

DATA_PATH = "/home/zxj/Downloads/data/models/"


def data_loader_creater(glove_path, embedding_path):
    def load_dataset(text_path=None, text_data=()):
        train_set = TextDataset(glove_path=glove_path, embedding_path=embedding_path,
                                text_path=text_path, text_data=text_data)
        train_loader = DataLoader(dataset=train_set, batch_size=128,
                                  shuffle=False, num_workers=4, pin_memory=False)
        return train_loader

    return load_dataset


def random_sample(percentage):
    def sample_vectors(vectors):
        length = len(vectors)
        random.shuffle(vectors)
        train_length = int(length*percentage)
        return vectors[:train_length], vectors[train_length:]

    return sample_vectors


def divide_list(input_list):
    first_column = [element[0] for element in input_list]
    second_column = [element[1] for element in input_list]
    return first_column, second_column


def classfication(params):
    train_and_dev = read_file(params["train_path"], lambda a: json.loads(a))
    train_x, train_y = unfold_domain(train_and_dev)
    train_and_dev = [ele for ele in zip(train_x, train_y)]
    sampler = random_sample(0.9)
    loader_factory = data_loader_creater(params["glove_path"], params["train_embedding"])
    train, dev = sampler(train_and_dev)
    train_data = loader_factory(text_data=divide_list(train))
    dev_data = loader_factory(text_data=divide_list(dev))
    test_data = data_loader_creater(params["glove_path"],
                                    params["test_embedding"])(text_path=params["test_path"])

    for x, y in test_data:
        print(x, y)

if __name__ == '__main__':
    root_path = "/Users/zxj/Google 云端硬盘/models_and_sample/"
    params = {"train_path": root_path + "en-ud-train-samples.txt",
              "test_path": root_path + "en-ud-test-samples.txt",
              "glove_path": root_path + "glove_train_and_test.txt",
              "train_embedding": root_path + "infer-sent-embeddings-train.npy",
              "test_embedding": root_path + "infer-sent-embeddings-test.npy"}

    classfication(params)