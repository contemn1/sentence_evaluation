import sys
import logging
import json
import numpy as np


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
                word_dict[word.lower()] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    return word_dict


def get_glove(glove_path, word_dict):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word.lower() in word_dict:
                word_vec[word.lower()] = vec

    print('Found {0}(/{1}) words with glove vectors'.format(
                    len(word_vec), len(word_dict)))

    return word_vec


def load_numpy_arraies(file_path):
    return np.load(file_path)