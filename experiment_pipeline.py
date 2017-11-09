# -*- coding: utf-8 -*-

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
import multiprocessing

DATA_PATH = "/home/zxj/Downloads/data/models/"


def data_loader_creater(glove_path, embedding_path):
    def load_dataset(text_path=None, text_data=(), batch_size=128, pin_memory=False):
        train_set = TextDataset(glove_path=glove_path, embedding_path=embedding_path,
                                text_path=text_path, text_data=text_data)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                                  shuffle=False, num_workers=multiprocessing.cpu_count(),
                                  pin_memory=pin_memory)
        return train_loader

    return load_dataset


def random_sample(percentage):
    def sample_vectors(vectors):
        length = len(vectors)
        random.shuffle(vectors)
        train_length = int(length*percentage)
        return vectors[:train_length], vectors[train_length:]

    return sample_vectors


def list_to_tuple(input_list):
    return int(input_list[0]), [input_list[1], input_list[2]]


def read_and_preprocess(path):
    data = read_file(path, lambda a: a.split("\t"))
    data_x = [list_to_tuple(ele) for ele in data]
    data_y = [int(ele[3]) for ele in data]
    return data_x, data_y


def classification(params):
    batch_size = 128 if "batch_size" not in params else params["batch_size"]
    loader_factory = data_loader_creater(params["glove_path"], params["train_embedding"])
    train_x, train_y = read_and_preprocess(params["train_path"])

    train_data = loader_factory(text_data=(train_x, train_y),
                                batch_size=batch_size,
                                pin_memory=params["cudaEfficient"])

    dev_data = loader_factory(text_data=read_and_preprocess(params["dev_path"]),
                              batch_size=batch_size,
                              pin_memory=params["cudaEfficient"]
                              )

    test_data = data_loader_creater(params["glove_path"],
                                    params["test_embedding"])(text_path=params["test_path"],
                                                              batch_size=batch_size,
                                                              pin_memory=params["cudaEfficient"])
    
    params["dimension"] = train_data.dataset.data_dimension()
    print(params["dimension"])
    classifier = BinaryClassifierEval(train=train_data, dev=dev_data, test=test_data)
    res = classifier.run(params=params)


if __name__ == '__main__':
    params = IOUtil.read_configs("/home/zxj/Documents/evaluation_parameters.ini")
    classification(params)
