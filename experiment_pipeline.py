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


def divide_list(input_list):
    first_column = [element[0] for element in input_list]
    second_column = [element[1] for element in input_list]
    return first_column, second_column


def classfication(params):
    train_and_dev = read_file(params["train_path"], lambda a: json.loads(a))
    train_x, train_y = unfold_domain(train_and_dev)
    train_and_dev = [ele for ele in zip(train_x, train_y)]
    sampler = random_sample(0.9)
    batch_size = 128 if "batch_size" not in params else params["batch_size"]
    loader_factory = data_loader_creater(params["glove_path"], params["train_embedding"])
    train, dev = sampler(train_and_dev)
    train_data = loader_factory(text_data=divide_list(train),
                                batch_size=batch_size,
                                pin_memory=params["cudaEfficient"])

    dev_data = loader_factory(text_data=divide_list(dev),
                              batch_size=batch_size,
                              pin_memory=params["cudaEfficient"]
                              )

    test_data = data_loader_creater(params["glove_path"],
                                    params["test_embedding"])(text_path=params["test_path"],
                                                              batch_size=batch_size,
                                                              pin_memory=params["cudaEfficient"])

    classifier = BinaryClassifierEval(train=train_data, dev=dev_data, test=test_data)
    res = classifier.run(params=params)
    print(res)


if __name__ == '__main__':
    params = IOUtil.read_configs("/Users/zxj/Google 云端硬盘/evaluation_parameters.ini")
