#!/usr/bin/python3
import logging
import sys
import spacy
from functools import reduce


def read_file(file_path, preprocess=lambda x: x):
    try:
        with open(file_path, encoding="utf8") as file:
            for sentence in file.readlines():
                yield (preprocess(sentence))

    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)


def load_sick(sick_path="/Users/samuel/Downloads/SICK/SICK.txt"):
    file_list = read_file(sick_path)
    file_list = (ele.split("\t")[1:7] for ele in file_list if not ele.startswith("pair_ID"))
    file_list = ([ele[0], ele[1], ele[2]] for ele in file_list)
    return file_list


def contain_no_connectives(sentence):
    if sentence.find('who') == -1 \
            and sentence.find('what') == -1 \
            and sentence.find('that') == -1 \
            and sentence.find('whether') == -1 \
            and sentence.find('where') == -1 \
            and sentence.find('how') == -1 \
            and sentence.find('when') == -1 \
            and sentence.find('which') == -1 \
            and sentence.find('why') == -1 \
            and sentence.find('whom') == -1 \
            and sentence.find('if') == -1 \
            and sentence.find('there') == -1 \
            and sentence.find('and') == -1:  # compound sentence
        return True
    else:
        return False


def find_object(language_model, sentence):
    doc = language_model(sentence)
    for token in doc:
        if token.dep_ == 'ROOT':
            for idx, child in enumerate(token.children):
                if child.dep_ == 'dobj':
                    return child.text
    return -1


def generate_negative_sentences(sentence):
    language_model = spacy.load('en')
    object_text = find_object(language_model, sentence)
    if object_text == -1:
        return -1
    sentence_list = sentence.split(' ')
    # if sentence_list[0] == 'A' or sentence_list[0] == 'An':
    #    sentence_list[0] = 'No'
    if 'is' in sentence_list or 'are' in sentence_list or 'am' in sentence_list:
        if 'is' in sentence_list:
            subject_idx = sentence_list.index('is')
        elif 'are' in sentence_list:
            subject_idx = sentence_list.index('are')
        elif 'am' in sentence_list:
            subject_idx = sentence_list.index('am')
        sentence_list.insert(subject_idx+1, 'not')
    else:
        return -1
    return " ".join(sentence_list).capitalize()


def filter_sick_dataset():
    sick_list = load_sick()
    language_model = spacy.load('en')
    filtered_sick_list = filter(lambda arr: sent_no_clause(language_model, arr[0])
                                and sent_no_clause(language_model, arr[1])
                                and contain_no_connectives(arr[0])
                                and contain_no_connectives(arr[1]),
                                sick_list)
    output_file = open("/Users/samuel/Desktop/passiveSentence.txt", "w")
    for ele in filtered_sick_list:
        passive_sentence = generate_negative_sentences(ele[0])
        if passive_sentence != -1:
            output_file.write(ele[0]+'\t'+ele[1]+'\t'+generate_negative_sentences(ele[0])+'\t'+ele[2]+'\n')
            print(ele[0]+'\t'+ele[1]+'\t'+generate_negative_sentences(ele[0])+'\t'+ele[2])
    output_file.close()


def sent_no_clause(language_model, sentence):
    doc = language_model(sentence)
    clause_set = {"ccomp", "csubj", "csubjpass", "xcomp", "iobj"}  # did not need sentence contains iobj.
    no_clause = True
    for token in doc:
        if token.dep_ == 'ROOT':
            children_dep = [child.dep_ for child in token.children]
            no_clause = reduce(lambda x, y: x and not y in clause_set,
                               children_dep, no_clause)
    return no_clause

filter_sick_dataset()

