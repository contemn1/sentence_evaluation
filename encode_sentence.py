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
import sent2vec

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


def read_file(file_path, encoding="utf-8", preprocess=lambda x: x):
    try:
        with open(file_path, encoding=encoding) as file:
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
    results = []
    embd = embd.reshape((-1, group_size, embd.shape[1]))
    for ele in embd:
        res_mat = cosine_similarity(ele)
        results.append([res_mat[0][1].item(), res_mat[0][2].item(), res_mat[1][2].item()])

    return np.array(results)


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


def power_mean(p):
    def calculate_mean(array):
        if p == 1:
            return np.mean(array, axis=0)

        mean = np.power(array, p).mean(axis=0).astype('complex')
        return np.power(mean, (1.0 / p)).real

    return calculate_mean


def get_embedding_from_glove(glove_path=GLOVE_PATH, power=1):
    def get_embedding(sentences):
        word_dict = get_word_dict(sentences)
        glove_dict = get_glove(glove_path, word_dict)
        sentences = [word_tokenize(sent) for sent in sentences]
        sentences = [[glove_dict[word] for word in sent if word in glove_dict] for sent in sentences]
        average = np.array([power_mean(1)(vec) for vec in sentences], dtype=np.float32)
        if power > 1:
            for index in range(2, power + 1):
                average = np.concatenate((average, [power_mean(index)(vec) for vec in sentences]), axis=1)

        return average

    return get_embedding


def get_embedding_from_infersent(model_path, batch_size=128, use_cuda=True):
    def get_infersent_embedding(sentences):
        model = resume_model(model_path, use_cuda=use_cuda)
        return model.encode(sentences, bsize=batch_size, tokenize=True, verbose=True)

    return get_infersent_embedding


def get_embedding_from_sent2vec(model_path):
    def get_fast_text_embedding(sentences):
        model = sent2vec.Sent2vecModel()
        model.load_model(model_path)
        embeddings = model.embed_sentences(sentences)
        return embeddings

    return get_fast_text_embedding


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
    return [snli_dict["sentence1"], snli_dict["sentence2"], snli_dict["gold_label"]]


def encode_triples(file_path, delimiter, triple_to_embedding):
    triplets = sentences_unfold(file_path=file_path, delimiter=delimiter)
    triplets = [ele.strip() for ele in triplets]
    embeddings = triple_to_embedding(triplets)
    return embeddings


def output_results(embeddings, verbose=False):
    results = calculate_pairwise_similarity(embeddings)
    score_array = np.mean(axis=0, a=results)
    if verbose:
        np.savetxt("average_scores", results)
    true_results = results[:, 0] > results[:, 1]
    accuracy = np.sum(true_results) / len(true_results)

    result_arr = score_array.tolist()
    result_arr.append(accuracy)
    result_arr = ["{:.2f}\\%".format(ele * 100) for ele in result_arr]
    return result_arr


if __name__ == '__main__':
    triple_path = "/home/zxj/Dropbox/typos/3typo.txt"
    model_path = DATA_PATH + "/infersent.allnli.pickle"
    sent2vec_model_path = "/media/zxj/New Volume/wiki_unigrams.bin"
    triples = sentences_unfold(file_path=triple_path, delimiter="\t")

    glove_result = output_results(get_embedding_from_glove(GLOVE_PATH, power=1)(triples))
    print("Glove Avg \t &　{0} \t \\\\ ".format("\t &".join(glove_result)))
    '''
    p_means_result = output_results(get_embedding_from_glove(GLOVE_PATH, power=3)(triples))
    print("P Means \t &　{0} \t \\\\ ".format("\t & ".join(p_means_result)))

    sent2vec_result = output_results(get_embedding_from_sent2vec(sent2vec_model_path)(triples), verbose=True)
    print("Sent2Vec \t &　{0} \t \\\\ ".format("\t & ".join(sent2vec_result)))
    '''
    infer_sent_result = output_results(get_embedding_from_infersent(model_path)(triples))
    print("Infersent \t &　{0} \t   \\\\ \\bottomrule".format("\t & ".join(infer_sent_result)))
