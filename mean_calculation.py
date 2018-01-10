import numpy as np
from encode_sentence import read_file
import json
from nltk.corpus import wordnet as wn

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


if __name__ == '__main__':
    syn("type")