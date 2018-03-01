import numpy as np
import json
from nltk.corpus import wordnet as wn
from encode_sentence import read_file
import spacy


def load_json_and_extract(json_string):
    json_dict = json.loads(json_string)

    return [json_dict["sentence1"], json_dict["sentence2"], json_dict["gold_label"]]

def syn(word):
    for net1 in wn.synsets(word):
        print(net1.definition(), net1.lemma_names())

def test():
    file_path = "/Users/zxj/Google 云端硬盘/experiment-results/SICK/contradiction-sentences.txt"
    a = read_file(file_path, preprocess=lambda x: x.split("\t")[:2])
    a = filter(lambda x: "n't" in x[0] or "n't" in x[1], a)
    for ele in a:
        result = ele[1] + "\001" + ele[0] + "\001" + ele[1] if "n't" in ele[0] \
            else ele[0] + "\001" + ele[1] + "\001" +ele[0]
        print(result)


def calculate_mean(file_path):
    scores = read_file(file_path, preprocess=lambda x: x.split("\t"))
    scores = [[float(ele) for ele in score_list] for score_list in scores]
    score_array = np.mean(axis=0, a=scores)
    print(score_array)


def calculate_score(file_path):
    scores = read_file(file_path, preprocess=lambda x: x.split("\t"))
    scores = [[float(ele) for ele in score_list] for score_list in scores]
    score_array = np.array(scores)
    res1 = score_array[:, 0] < score_array[:, 1]
    res2 = score_array[:, 2] < score_array[:, 1]
    res3 = np.zeros(len(score_array))
    for index in range(len(score_array)):
        res3[index] = res1[index] and res2[index]
    print(np.sum(res3) / len(res3))

def divide(sent_path, score_path):
    sents = read_file(sent_path, preprocess=lambda x: x.split("\002")[0])
    scores = read_file(score_path, preprocess=lambda x: x.split("\t"))
    scores = [[float(ele) for ele in score_list] for score_list in scores]
    score_array = np.array(scores)
    en_model = spacy.load("en")
    index = []
    reverse_index = []
    for idx, ele in enumerate(sents):
        res = en_model(ele)
        for token in res:
            if token.dep_ == "ccomp" and token.idx < token.head.idx:
                reverse_index.append(idx)
                break
            else:
                index.append(idx)
                break

    first = score_array[index]
    res_frist = first[:, 0] > first[:, 2]

    second = score_array[reverse_index]
    res_second = second[:, 0] > second[:, 2]

    print(first.mean(axis=0))
    print(res_frist.sum() / len(res_frist), len(res_frist))

    print(second.mean(axis=0))
    print(res_second.sum() / len(res_second), len(res_second))



if __name__ == '__main__':
    root_path = "/Users/zxj/Google 云端硬盘/experiment-results/negation_of_quantifier/"
    files = ["quantifier_negation_glove.txt",
             "quantifier_negation_infersent.txt",
             "quantifier_negation_skipthoughts.txt"]
    file_paths = [root_path + ele for ele in files]
    for ele in file_paths:
        (calculate_score(ele))