from models import BLSTMEncoder
import numpy as np
import logging
import json
import sys
import torch
from sklearn.metrics.pairwise import cosine_similarity
from IOUtil import get_word_dict
from IOUtil import get_glove
from nltk.tokenize import word_tokenize
from functools import reduce
import spacy
from nltk.corpus import wordnet as wn

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


def read_file(file_path, preprocess=lambda x: x):
    try:
        with open(file_path, encoding="utf8") as file:
            for sentence in file.readlines():
                yield (preprocess(sentence))

    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)


def resume_model(model_path, glove_path=GLOVE_PATH, use_cuda=False):
    location_function = None if use_cuda else lambda storage, loc: storage
    model = torch.load(model_path, map_location=location_function)  # type: BLSTMEncoder
    model.set_glove_path(glove_path)
    model.build_vocab_k_words(K=100000)
    return model


def encoding_setences(model_path, glove_path, sentence_list, use_cuda=False) -> np.ndarray:
    model = resume_model(model_path, glove_path,use_cuda)  #type: BLSTMEncoder
    embeddings = model.encode(sentence_list, bsize=128, tokenize=True, verbose=True)
    return embeddings

def calculate_pairwise_similarity(embd, group_size=3):
    embd = embd.reshape((-1, group_size, embd.shape[1]))
    for ele in embd:
        res_mat = cosine_similarity(ele)
        res_need = [str(res_mat[0][1].item()), str(res_mat[1][2].item()), str(res_mat[0][2].item())]
        print("\t".join(res_need))


def sentences_unfold(file_path, delimiter="\001", predicate=lambda x: len(x) == 3):
    sent_list = read_file(file_path, preprocess=lambda ele: ele.split(delimiter))
    sent_list = [arr for arr in sent_list if predicate(arr)]
    sent_list = [ele for arr in sent_list for ele in arr]
    return sent_list


def load_sick(sick_path="/Users/zxj/Downloads/SICK/SICK.txt"):
    file_list = read_file(sick_path)
    file_list = (ele.split("\t")[1:7] for ele in file_list if not ele.startswith("pair_ID"))
    file_list = ([ele[0], ele[1], ele[2]] for ele in file_list)
    return file_list


def encode_sick(use_cuda, model_path):
    file_list = load_sick()
    file_list = list(file_list)
    first = [ele[0] for ele in file_list]
    second = [ele[1] for ele in file_list]
    third = [ele[2] for ele in file_list]
    model = resume_model(model_path, use_cuda=use_cuda)
    first_emb = model.encode(first, bsize=128, tokenize=True, verbose=True)
    second_emb = model.encode(second, bsize=128, tokenize=True, verbose=False)
    res = cosine_similarity(first_emb, second_emb).diagonal().tolist()
    for ele in zip(res, third):
        print(str(ele[0]) + "\t" + ele[1])


def get_embedding_from_glove(glove_path=GLOVE_PATH):
    def get_embedding(sentences):
        word_dict = get_word_dict(sentences)
        glove_dict = get_glove(glove_path, word_dict)
        sentences = [word_tokenize(sent) for sent in sentences]
        sentences = [np.mean([glove_dict[word] for word in sent if word in glove_dict], axis=0) for sent in sentences]
        sentences = np.array(sentences)
        return sentences

    return get_embedding


def get_embedding_from_infersent(model_path, batch_size=128, use_cuda=True):
    def get_infersent_embedding(sentences):
        model = resume_model(model_path, use_cuda=use_cuda)
        return model.encode(sentences, bsize=batch_size, tokenize=True, verbose=True)

    return get_infersent_embedding


def sent_no_clause(language_model, sentence):
    doc = language_model(sentence)
    clause_set = {"ccomp", "csubj", "csubjpass", "xcomp"}
    no_clause = True
    for token in doc:
        if token.dep_ == 'ROOT':
            children_dep = [child.dep_ for child in token.children]
            no_clause = reduce(lambda x, y: x and not y in clause_set,
                               children_dep, no_clause)
    return no_clause


def filter_sick_dataset():
    sick_list =load_sick()
    language_model = spacy.load('en')
    filtered_sick_list = filter(lambda arr: sent_no_clause(language_model, arr[0])
                                and sent_no_clause(language_model, arr[1]),
                                sick_list)
    for ele in filtered_sick_list:
        print(ele)


def pipeline():
    file_path = "inversion_tuple.txt"
    model_path = "/home/zxj/Downloads/data/infersent.allnli.pickle"
    sentences = list(read_file(file_path, preprocess=lambda x: x.split("\001")))
    first = [tup[0] for tup in sentences]
    second = [tup[1].strip() for tup in sentences]
    model = resume_model(model_path, use_cuda=True)
    first_emb = model.encode(first, bsize=128, tokenize=True, verbose=True)
    second_emb = model.encode(second, bsize=128, tokenize=True, verbose=False)
    res = cosine_similarity(first_emb, second_emb).diagonal().tolist()
    for ele in res:
        print(ele)


def take_two(sentence):
    triple = sentence.split("\t")
    return triple[0], triple[2]


def filter_passive(file_path):

    sent_dict = {}
    passive_tuples = read_file(file_path, preprocess=take_two)
    for ele in passive_tuples:
        active, passive = ele
        if active not in sent_dict:
            sent_dict[active] = passive

    for key, value in sent_dict.items():
        print(key + "\t" + value)


def process_snli_json(snli_json):
    snli_dict = json.loads(snli_json)
    return [snli_dict["sentence1"], snli_dict["sentence2"]]


def encode_triples(file_path, delimiter, triple_to_embedding):
    triplets = sentences_unfold(file_path=file_path, delimiter=delimiter)
    triplets = [ele.strip() for ele in triplets]
    embeddings = triple_to_embedding(triplets)
    calculate_pairwise_similarity(embeddings)

if __name__ == '__main__':
    triple_path = "/home/zxj/Dropbox/data/sentence_triples_random.txt"
    model_path = DATA_PATH + "/infersent.allnli.pickle"
    encode_triples(triple_path,
                   delimiter="\t",
                   triple_to_embedding=get_embedding_from_glove(GLOVE_PATH))
