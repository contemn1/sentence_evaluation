import re
import spacy
import random
import numpy as np
import operator
from typing import Iterator
from typing import List

from encode_sentence import read_file
from encode_sentence import take_two
from IOUtil import output_list_to_file
from nltk.corpus import wordnet as wn

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


def select_similar_sentence(sents: Iterator[List[str]]):
    for arr in sents:
        if negation.search(arr[0]) or negation.search(arr[1]):
            continue

        first = arr[0].split(" ")
        second = arr[1].split(" ")
        dist = levenshtein(first, second)
        similarity = 1 - dist / ((len(first) + len(second)) / 2.0)
        if dist == 1 or 1 > similarity > 0.85:
            yield (arr[0] + "\t" + arr[1])


def reorder_with_pivot(index):
    def reorder_list(arr):
        return " ".join(arr[index:]) + " " + " ".join(arr[:index])

    return reorder_list


def reorder_randomly(arr):
    return " ".join(random.sample(arr, len(arr)))


def generate_random(file_path, get_reordered):
    """

    :param file_path: specify path of a certain file
    :return:
    """
    sent_tuple = read_file(file_path, lambda x: x.strip().split("\t"))
    for arr in sent_tuple:
        first_sent = arr[0]
        first_arr = first_sent.split(" ")
        reversed_first = get_reordered(first_arr)
        arr.append(reversed_first)
        print("\t".join(arr).strip())


def load_sick2(sick_path="/Users/zxj/Downloads/SICK/SICK.txt"):
    file_list = read_file(sick_path)
    file_list = (ele.split("\t")[1:7] for ele in file_list if not ele.startswith("pair_ID"))
    file_list = ([ele[0], ele[1], ele[2], ele[3]] for ele in file_list)
    return file_list


def calculate_word_frequency():
    file_path = "/Users/zxj/Desktop/snli_1.0/possible_contradiction"
    sent_tuple = read_file(file_path, lambda x: x.split("\t"))
    results = {}
    for arr in sent_tuple:
        arr_tuple = zip(arr[0].split(" "), arr[1].split(" "))
        arr_tuple = [ele for ele in arr_tuple if ele[0] != ele[1]]
        arr_str = (arr_tuple[0][0] + "\t" + arr_tuple[0][1]).strip().lower()
        if arr_str not in results:
            results[arr_str] = 1
        else:
            results[arr_str] += 1

    for key, value in results.items():
        if value == 1:
            print(key, value)


def find_antonomy(word, tag):
    def process_word(input):
        return wn.synsets(input, pos=tag) if tag else wn.synsets(word)

    lemmas = (l for syn in process_word(word) for l in syn.lemmas())
    antonomy = [anto.name() for l in lemmas for anto in l.antonyms() if anto]
    antonomy_set = set(antonomy)
    if not antonomy_set:
        return word
    else:
        return "|".join(antonomy_set)


def convert_postag(treebank_tag: str):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''


def find_pos_of_certain_word(model, sentence, word):
    for ele in model(sentence):
        if ele.text == word:
            new_word = ele.text if ele.pos_ != "VERB" else ele.lemma_
            return new_word, convert_postag(ele.tag_)

    return word, ""


def devide_dataset():
    sents = load_sick2()
    sents = (ele for ele in sents if float(ele[3]) > 4.6 and ele[2].lower() == "entailment")
    model = spacy.load('en')
    res1 = []
    res2 = []
    for ele in select_similar_sentence(sents):
        arr = ele.split("\t")
        first = arr[0].split(" ")
        second = arr[1].split(" ")
        if len(first) < len(second):
            first, second = second, first
            arr[0], arr[1] = arr[1], arr[0]

        diff = set(first) - set(second)
        if len(diff) == 1:
            word, pos = find_pos_of_certain_word(model, arr[0], "".join(diff))
            antonomy = find_antonomy(word, pos)
            third = re.sub(word, antonomy, arr[0])
            if arr[0] != third:
                res1.append("\t".join([arr[0], arr[1], third]))
            else:
                res2.append("\t".join([arr[0], arr[1], third]))

    output_list_to_file("changed.txt", res1)
    output_list_to_file("unchanged.txt", res2)


def exists_clause(model, sent):
    for ele in model(sent):
        if ele.dep_ == "ccomp":
            return ele.head.text, sent

    return ''


if __name__ == '__main__':
    sents = read_file("SICK_test_antonym.txt", preprocess=lambda x: x.strip().split("\t"))
    label2id = {'CONTRADICTION': 0, 'NEUTRAL': 1, 'ENTAILMENT': 2}
    for index, tuple in enumerate(sents):
        if tuple[4] not in label2id:
            print(index, tuple)