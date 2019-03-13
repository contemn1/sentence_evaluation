# -*- coding: utf-8 -*-

import configparser
import logging
import re
import sys

import numpy as np
from gensim.models import KeyedVectors
from nltk.tokenize import sent_tokenize

THINKREGEX = re.compile(" says? | said | knows? | knew | thinks? | thought")


def get_word_dict(sentences, tokenize=True):
    # create vocab of words
    word_dict = {}
    if tokenize:
        from nltk.tokenize import word_tokenize
    sentences = [s.split() if not tokenize else word_tokenize(s)
                 for s in sentences]
    for sent in sentences:
        for word in sent:
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    return word_dict


def get_glove(glove_path, word_dict):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.fromstring(vec.strip(), sep=' ')

    logging.info('Found {0}(/{1}) words with glove vectors'.format(
        len(word_vec), len(word_dict)))

    return word_vec


def get_glove_dict(glove_path):
    word_vec = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            word, vec = line.split(' ', 1)
            word_vec[word] = np.fromstring(vec.strip(), sep=' ')

    return word_vec


def load_numpy_arraies(file_path):
    return np.load(file_path)


def unfold_domain(text_list, keys=frozenset(["positive", "negative"])):
    sample_list = [[(index, text_dict[key].split(",")) for index, text_dict in enumerate(text_list)] for key in keys]

    all_tuples = []
    for sub_list in sample_list:
        for pair in sub_list:
            new_list = [(pair[0], ele.split("->")) for ele in pair[1]]
            all_tuples.extend(new_list)

    train_x = [tup for tup in all_tuples if len(tup[1]) > 1]
    train_y = [0] * len(sample_list[0]) + [1] * len(sample_list[1])
    return train_x, train_y


def output_list_to_file(file_path, output_list, process=lambda x: x):
    try:
        with open(file_path, mode="w+", encoding="utf-8") as file:
            for line in output_list:
                file.write(process(line))
                file.write("\n")
    except IOError as error:
        logging.error("Failed to open file {0}".format(error))


def read_configs(config_path):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_path, encoding="utf-8")
    config_dict = {key: string_to_attributes(value) for key, value in config["arguments"].items()}
    return config_dict


def load_pretrained_word2vec(model_path, binary=True):
    return KeyedVectors.load_word2vec_format(fname=model_path, binary=binary)


def string_to_attributes(input_string):
    if input_string.lower() in {"yes", "true"}:
        return True
    if input_string.lower() in {"no", "false"}:
        return False

    if input_string.isdigit():
        return int(input_string)

    return input_string


def read_text_file_with_think(input_path):
    sentecnes = []
    try:
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                results = sent_tokenize(line)
                sents = [ele1 for ele1 in results if THINKREGEX.search(ele1)]
                for ele in sents:
                    if len(ele.split(" ")) < 30:
                        print(ele)

            return sentecnes
    except IOError as err:
        print("Failed to read file {0}".format(err))
        return sentecnes


def read_file(file_path, encoding="utf-8", preprocess=lambda x: x):
    try:
        with open(file_path, encoding=encoding) as file:
            for sentence in file.readlines():
                yield (preprocess(sentence))

    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)


if __name__ == '__main__':
    path = "/home/zxj/Downloads/new_corpus/argument_compositionality.txt"
    argument = read_file(path, preprocess=lambda x: x.strip().split("\t"))
    result_list = []
    for ele in argument:
        if len(ele) == 3:
            result_list.append([ele[1], ele[0], ele[-1]])

    output_list_to_file("/home/zxj/Downloads/new_corpus/new_argument_compositionality.txt", result_list, process=lambda x: "\t".join(x))