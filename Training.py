from customized_dataset import EmbeddingDataset
from encode_sentence import load_sick
from encode_sentence import read_file
from encode_sentence import get_embedding_from_infersent
from encode_sentence import get_embedding_from_glove
from sklearn.model_selection import train_test_split
from IOUtil import output_list_to_file
import numpy as np
from torch.utils.data import DataLoader
from tools.new_classifier import MLP
import multiprocessing
import torch

def dataset_split(data, train_percent, valid_percent):
    data_train, data_valid_test = train_test_split(data, train_size=train_percent)
    data_valid, data_test = train_test_split(data_valid_test, test_size=valid_percent)
    return data_train, data_valid, data_test


def split_sick_set(sick_path, train_path, valid_path, test_path):
    triples = list(load_sick(sick_path))
    train, valid, test = dataset_split(triples, 0.8, 0.5)
    string_format = lambda x: "\t".join(x).strip()
    output_list_to_file(train_path, train, string_format)
    output_list_to_file(valid_path, valid, string_format)
    output_list_to_file(test_path, test, string_format)


def load_data(path, encode_sents):
    file_list = read_file(path, lambda x: x.strip().split("\t")[1:])
    filter(lambda ele: not ele.startswith("pair_ID"), file_list)
    file_list = ([ele[0], ele[1], ele[3]] for ele in file_list)
    tag_map = {"ENTAILMENT": 0,
               "NEUTRAL": 1,
               "CONTRADICTION": 2}
    scores = []
    first_sents = []
    second_sents = []
    for arr in file_list:
        if arr[2] in tag_map:
            scores.append(tag_map[arr[2]])
            first_sents.append(arr[0])
            second_sents.append(arr[1])

    embeddings_a = encode_sents(first_sents)
    embeddings_b = encode_sents(second_sents)
    return np.c_[np.abs(embeddings_a - embeddings_b), embeddings_a * embeddings_b], scores


def create_data_loader(data_x, data_y, params):
    dataset = EmbeddingDataset(data_x, data_y)
    loader = DataLoader(dataset=dataset, batch_size=params["batch_size"],
                              shuffle=True, num_workers=multiprocessing.cpu_count(),
                              pin_memory=params["use_cuda"])
    return loader


if __name__ == '__main__':
    SICK_ROOT = "/home/zxj/Downloads/sick/SICK/"
    DATA_PATH = "/home/zxj/Downloads/data"
    GLOVE_PATH = DATA_PATH + "/glove.840B.300d.txt"
    model_path = DATA_PATH + "/infersent.allnli.pickle"
    encode_function = get_embedding_from_infersent(model_path, batch_size=128, use_cuda=True)
    train_path = SICK_ROOT + "SICK_train.txt"
    dev_path = SICK_ROOT + "SICK_trial.txt"
    test_path = SICK_ROOT + "SICK_test_annotated.txt"
    data_test, labels_test = load_data(test_path, encode_function)
    params = {"batch_size": 64,
              "use_cuda": True,
              "nhid": 16,
              "max_epoch": 200,
              "epoch_size": 4,
              "tenacity": 10}
    test_loader = create_data_loader(data_test, labels_test, params)
    classifier = MLP(params, data_test.shape[1], nclasses=3, l2reg=10**-5)
    classifier.model.load_state_dict(torch.load("models/infersent_sick.pt"))
    print(classifier.score(test_loader))
    torch.save(classifier.model.state_dict(), "infersent_classifier")