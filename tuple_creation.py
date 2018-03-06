import re
import spacy
import random
import numpy as np
from typing import Iterator

from encode_sentence import read_file
from encode_sentence import load_sick

NO_PATTERN = re.compile("No")
CURRENT_PATTERN = re.compile("is|are|am")
negation = re.compile("not|n\'t")


def negate_quantifier(input_sent: str):
    res = CURRENT_PATTERN.search(input_sent)
    if not res:
        return input_sent
    else:
        without_verb = CURRENT_PATTERN.sub("", input_sent)
        change_start = NO_PATTERN.sub("There {0} no".format(res.group()), without_verb)
        return re.sub("\s+", " ", change_start)


def negate_verb(input_sent: str):
    res = CURRENT_PATTERN.search(input_sent)
    if not res:
        return input_sent
    else:
        return CURRENT_PATTERN.sub(res.group() + " not", input_sent)


def generate_negative_samples():
    file_path = "/Users/zxj/PycharmProjects/sentence_evaluation/dataset/negative_no_unique.txt"
    file_generator = read_file(file_path, preprocess=lambda x: x.split("\t"))
    file_generator = ([tup[0], tup[1].strip(), negate_verb(tup[0])] for tup in file_generator)
    for ele in file_generator:
        print("\001".join(ele).strip())

def extract_verb(language_model, sentence):
    result_list = language_model(sentence)
    return_str = ""
    for ele in result_list:
        if ele.pos_ == "VERB":
            return_str = ele.text
            for child in ele.children:
                if child.dep_ == "prep":
                    return_str += " " + child.text

    return return_str


def extrac_verb_phase(language_model, sentence):
    result_list = language_model(sentence)
    for ele in result_list:
        if ele.dep_ == "aux":
            root = ele.head
            lemma = root.lemma_
            pattern = re.compile("{0}[\w\s]+".format(root))
            res = pattern.search(sentence)
            if res:
                result = res.group()
                return re.sub(root.text, lemma, result)
    return ""


def generate_verb_phases():
    language_model = spacy.load('en')
    sents = read_file("test2.txt",
                      preprocess=lambda x: extrac_verb_phase(language_model, x))
    sents = (ele for ele in sents if len(ele.split(" ")) > 1)
    for ele in sents:
        print(ele.strip())


def generate_index(index_set):
    def random_once(lower_bound, upper_bound):
        current = random.randint(lower_bound, upper_bound)
        if current not in index_set:
            index_set.add(current)
            return current
        else:
            random_once(lower_bound, upper_bound)

    return random_once

def past_particle(word):
    if word and  word[-1] == "e":
        return word + "d"

    if word == "rob":
        return "robbed"

    else:
        return word + "ed"


def simple_triple(names_list, verbs_list, times):
    index_set = set()
    index_generator = generate_index(index_set)
    sent_template = "{0} {1}s {2}"
    for verb in verbs_list:
        for _ in range(times):
            first = names_list[index_generator(0, 199)]
            second = names_list[index_generator(0, 199)]
            active_sent = sent_template.format(first, verb, second)
            passive_sent = "{0} is {1} by {2}".format(second, past_particle(verb), first)
            active_inversion = sent_template.format(second, verb, first)
            print(active_sent + "\t" + passive_sent + "\t" + active_inversion)

        index_set.clear()


def complicate_triple(names_list, verbs_list, phase_list, times):
    index_set = set()
    index_generator = generate_index(index_set)
    pattern = re.compile("sb")
    for verb in verbs_list:
        for _ in range(int(times / 3)):
            first = names_list[index_generator(0, 199)]
            second = names_list[index_generator(0, 199)]
            for _ in range(3):
                third = phase_list[index_generator(0, len(phase_list)-1)]
                active_sent = first + " " + pattern.sub(second, verb) + " " + third
                passive_sent = "{0} is {1} to {2}".format(second,
                                                      past_particle(verb.split(" ")[0]),
                                                      third)

                active_inversion = second + " " + pattern.sub(first, verb) + " " + third
                print(active_sent + "\t" + passive_sent + "\t" + active_inversion)
                index_set.clear()

            index_set.clear()


def equal_distance(str1, str2):
    if str1 == str2:
        return 0
    else:
        return 1


def levenshtein(sent1, sent2, str_distance=equal_distance):
    if len(sent1) < len(sent2):
        return levenshtein(sent2, sent1)

    # len(s1) >= len(s2)
    if len(sent2) == 0:
        return len(sent1)

    previous_row = range(len(sent2) + 1)
    for i, c1 in enumerate(sent1):
        current_row = [i + 1]
        for j, c2 in enumerate(sent2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + str_distance(c1, c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def select_similar_sentence(sents: Iterator[str]):
    for arr in sents:
        if negation.search(arr[0]) or negation.search(arr[1]):
            continue

        first = arr[0].split(" ")
        second = arr[1].split(" ")
        tag = arr[2]
        dist = levenshtein(first, second)
        similarity = 1 - dist / ((len(first) + len(second)) / 2.0)
        if tag == "CONTRADICTION" and (dist == 1 or similarity > 0.85):
            print("\t".join(arr[:-1]))


if __name__ == '__main__':
    file_path = "/Users/zxj/Google 云端硬盘/experiment-results/similar_sentences_new.txt"
    sent_tuple = read_file(file_path, lambda x: x.strip().split("\t"))
    fixed_index = 3
    for arr in sent_tuple:
        first_sent = arr[0]
        first_arr = first_sent.split(" ")
        reversed_first = " ".join(first_arr[fixed_index:]) + " ".join(first_arr[:fixed_index])
        arr.append(reversed_first)
        print("\t".join(arr).strip())