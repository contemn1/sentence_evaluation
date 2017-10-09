import logging
import sys
import json
import IOUtil
import numpy as np
import re

DATAPATH = "/Users/zxj/Google 云端硬盘/models_and_sample/"
BRACKETS = re.compile("[\[\]]")


def read_file(file_path, preprocess):
    content_list = []
    try:
        with open(file_path, encoding="utf8") as file:
            contents = file.readlines()
            for sentence in contents:
                content_list.append(preprocess(sentence))

            return content_list
    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)


def get_glove_in_corpus(train_path, test_path):
    dicts_list = read_file(train_path, lambda x: json.loads(x))
    dicts_list_test = read_file(test_path, lambda x: json.loads(x))
    sentences = [dicts["sentence"] for dicts in dicts_list]
    sentences.extend([dicts["sentence"] for dicts in dicts_list_test])
    words_dict = IOUtil.get_word_dict(sentences)
    glove_dict = IOUtil.get_glove("/Users/zxj/Downloads/glove.840B.300d.txt", words_dict)
    for key, value in glove_dict.items():
        print(key + " " + value.strip("\n\s"))


def split_string(input):
    return input.split(" ", 1)


def get_glove_array(query, glove_dict):
    return glove_dict[query] if query in glove_dict else np.zeros(300, dtype=np.float32)


def sample_to_vectors(dicts_list, key, glove_dict, setence_vector):
    sample_list = dicts_list[key].split(",")
    all_tuples = [ele.split("->") for ele in sample_list if ele]
    valid_tuples = [tup for tup in all_tuples if len(tup) >= 2]
    word_vectors = [np.hstack((get_glove_array(ele[0], glove_dict),
                               get_glove_array(ele[1], glove_dict)))
                    for ele in valid_tuples]
    return [np.hstack((setence_vector, vector)) for vector in word_vectors]


def main(glove_path, train_path):
    glove_list = read_file(glove_path, split_string)
    glove_dict = {ele[0]: np.fromstring(ele[1], sep=" ") for ele in glove_list}
    dicts_list = read_file(train_path, lambda x: json.loads(x))
    setence_embedding_path = DATAPATH + "infer-sent-embeddings-test.npy"
    setence_embeddings = np.load(setence_embedding_path)
    all_positive_samples = []
    all_negative_samples = []
    for index in range(len(dicts_list)):
        ele = dicts_list[index]
        positive_vectors = sample_to_vectors(ele, "positive", glove_dict, setence_embeddings[index])
        negative_vectors = sample_to_vectors(ele, "negative", glove_dict, setence_embeddings[index])
        all_positive_samples.extend(positive_vectors)
        all_negative_samples.extend(negative_vectors)

    output_positive = "all_positive_samples_test"
    output_negative = "all_negative_samples_test"
    np.save(output_positive, all_positive_samples)
    np.save(output_negative, all_negative_samples)

