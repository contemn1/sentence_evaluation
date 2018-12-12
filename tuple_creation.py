import os
import random
import re
from typing import Iterator
from typing import List

import numpy as np
import spacy
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn

from IOUtil import output_list_to_file
from encode_sentence import read_file

NO_PATTERN = re.compile("No")
CURRENT_PATTERN = re.compile("is|are|am")
negation = re.compile("not|n\'t")


def negate_quantifier(input_sent: str):
    res = CURRENT_PATTERN.search(input_sent)
    if not res:
        return input_sent
    else:
        without_verb = CURRENT_PATTERN.sub("", input_sent)
        change_start = NO_PATTERN.sub("There {0} no".format(res.group()),
                                      without_verb)
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
    file_generator = ([tup[0], tup[1].strip(), negate_verb(tup[0])] for tup in
                      file_generator)
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
    if word and word[-1] == "e":
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
            passive_sent = "{0} is {1} by {2}".format(second,
                                                      past_particle(verb),
                                                      first)
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
                third = phase_list[index_generator(0, len(phase_list) - 1)]
                active_sent = first + " " + pattern.sub(second,
                                                        verb) + " " + third
                passive_sent = "{0} is {1} to {2}".format(second,
                                                          past_particle(
                                                              verb.split(" ")[
                                                                  0]),
                                                          third)

                active_inversion = second + " " + pattern.sub(first,
                                                              verb) + " " + third
                print(
                    active_sent + "\t" + passive_sent + "\t" + active_inversion)
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
    file_list = (ele.split("\t")[1:7] for ele in file_list if
                 not ele.startswith("pair_ID"))
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
    sents = (ele for ele in sents if
             float(ele[3]) > 4.6 and ele[2].lower() == "entailment")
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


def random_typo(src_list, typo_dict, times):
    history_typo = {}
    while len(history_typo) < times:
        random_word = random.choice(src_list)
        if random_word not in typo_dict or random_word in history_typo:
            continue

        history_typo[random_word] = random.choice(typo_dict[random_word])

    return history_typo


def generate_typos():
    a = read_file("/Users/zxj/Google 云端硬盘/similar_typos.txt",
                  preprocess=lambda x: x.strip().split("\t"))
    typo_dict = {arr[0]: arr[1].split(" ") for arr in a}
    num = 3
    sents = read_file("/Users/zxj/Dropbox/data/similar_structure.txt",
                      preprocess=lambda x: x.strip().split("\t"))
    for arr in sents:
        words = arr[0].split(" ")
        first = [1 if word in typo_dict else 0 for word in words]
        if np.sum(first) >= num:
            typo_map = random_typo(words, typo_dict, 1)
            for key, value in typo_map.items():
                new_sent = re.sub(key, value, arr[0])
                arr.append(new_sent)
                print("\t".join(arr))


def factual_test():
    sents = read_file(
        "/Users/zxj/Google 云端硬盘/experiment-results/Clause Relatedness/clause_relatededness_samples.txt",
        preprocess=lambda x: x.strip().split("\002")[:-1])
    replace_dict = {"say": "deny",
                    "says": "denies",
                    "said": "denied",
                    "think": "doubt",
                    "thinks": "doubts",
                    "thought": "doubted"}

    for arr in sents:
        for key, value in replace_dict.items():
            pattern = re.compile(" {0}".format(key))
            if not pattern.search(arr[0]):
                continue

            new_sent = pattern.sub(" {0}".format(value), arr[0])
            arr.append(new_sent)
            print("\t".join(arr))


def first_to_upper(sentence):
    sent_list = sentence.split(" ")
    return sent_list[0].title() + " " + " ".join(sent_list[1:])


def extract_clause(sent, language_model):
    result_list = language_model(sent)
    verb_in_clause = [ele for ele in result_list if
                      ele.dep_ == "ccomp" and ele.head.dep_ == "ROOT"]
    if not verb_in_clause:
        return ""

    if verb_in_clause[0].idx < verb_in_clause[0].head.idx:
        result = re.search('\"(.+?)\"', sent)
        return first_to_upper(result.group()[1:-1]) if result else ""
    subj_in_clause = [subj for subj in verb_in_clause[0].children if
                      subj.dep_ == "nsubj"]
    if not subj_in_clause:
        return ""

    if not subj_in_clause[0].children:
        return sent[subj_in_clause[0].idx:]

    sorted_children = sorted(list(subj_in_clause[0].children),
                             key=lambda x: x.idx)
    if not sorted_children:
        return ""
    start_index = sorted_children[0].idx
    result = sent[start_index:]
    return first_to_upper(result)


def extrac_caluse_from(language_model):
    path = "/Users/zxj/Downloads/dataset-sts/data/para/msr/msr-para-train.tsv"
    file = read_file(path, preprocess=lambda x: x.split("\t"))
    file = [(ele[3], ele[4].strip()) for ele in file if ele[0] == "1"]
    say_regex = re.compile("said|says?|saying|tell|told|thinks")
    for a, b in file:
        if say_regex.search(a):
            clause = extract_clause(a, language_model)
            if clause:
                print(a + "\t" + clause)
        if not say_regex.search(a) and say_regex.search(b):
            clause = extract_clause(b, language_model)
            if clause:
                print(b + "\t" + clause)


def negate_word_msr(old_sent, language_model):
    new_sent = old_sent
    for token in language_model(old_sent):
        if token.dep_ == "ccomp" and token.head.dep_ == "ROOT":
            neg_child = [child for child in token.children if
                         child.dep_ == "neg"]
            if neg_child:
                neg_word = neg_child[0]
                new_sent = re.sub("{0} ".format(neg_word), " ", old_sent)
                new_sent = re.sub(" wo ", " will ", new_sent)
                new_sent = re.sub(" does ", " ", new_sent)

            elif token.tag_ == "VBZ":
                if token.text == "is":
                    new_sent = re.sub(" is ".format(token.text),
                                      " isn't ",
                                      old_sent)
                else:
                    new_sent = re.sub("{0}".format(token.text),
                                      "doesn't {0}".format(token.lemma_),
                                      old_sent)

            elif token.tag_ == "VBP":
                new_sent = re.sub("{0}".format(token.text),
                                  "don't {0}".format(token), old_sent)

            elif token.tag_ == "VBD":
                new_sent = re.sub("{0}".format(token.text),
                                  "didn't {0}".format(token.lemma_), old_sent)

            else:
                aux_word = [child for child in token.children if
                            child.dep_ == "aux"]
                if aux_word and not aux_word[0].text == "will":
                    new_sent = re.sub(" {0} ".format(aux_word[0].text),
                                      " {0}n't ".format(aux_word[0].text),
                                      old_sent)
                if aux_word and aux_word[0].text == "will":
                    new_sent = re.sub(" {0} ".format(aux_word[0].text),
                                      " won't ".format(aux_word[0].text),
                                      old_sent)
            return new_sent


def negation_variant_sick(sentence_list):
    is_regex = re.compile(" is ")
    are_regex = re.compile(" are ")
    for ele in sentence_list:
        first_word = ele.split(" ")[0]
        if is_regex.search(ele):
            second = is_regex.sub(" is not ", ele)
            third = is_regex.sub(" ", ele)
            third = re.sub("^{0}".format(first_word), "There is no", third)
            print(ele + "\t" + second + "\t" + third)

        if are_regex.search(ele):
            second = are_regex.sub(" are not ", ele)
            third = are_regex.sub(" ", ele)
            third = re.sub("^{0}".format(first_word), "There are no", third)
            print(ele + "\t" + second + "\t" + third)


def extract_verb_phases(sent, model):
    doc = model(sent)
    verb_phases = []
    for ele in doc:
        if ele.dep_ == "ROOT" and ele.pos_ == "VERB":
            print(ele.pos_)
            verb_phases.append(ele.text)
            continue
        if ele.dep_ == "prep" and ele.head.dep_ == "ROOT" and ele.head.pos_ == "VERB":
            verb_phases.append(ele.text)
            continue
        if ele.dep_ == "prt" and ele.head.dep_ == "ROOT" and ele.head.pos_ == "VERB":
            verb_phases.append(ele.text)

    return " ".join(verb_phases)


def valid_sentence(sentence, parser):
    doc = parser(sentence)
    for token in doc:
        if token.dep_ == "conj" and token.pos_ == "NOUN" and token.head.pos_ == "NOUN":
            return True
    return False


def valid_structure(docs):
    for ele in docs:
        if ele.dep_ == "conj" and ele.head.dep_ in {"dobj", "pobj"}:
            return True
    return False


def generate_compositional_triplets(sentence, doc):
    sigular_set = {"NN", "NNP"}
    for ele in doc:
        if ele.dep_ == "conj" and ele.head.dep_ in {"dobj", "pobj"}:
            end_index = ele.idx + len(ele.text)
            start_element = [child for child in ele.head.children if child.dep_ == "cc"]
            if not start_element:
                continue
            start_index = start_element[0].idx
            first_part = sentence[:start_index - 1]
            second_part = sentence[end_index:]
            original_sentence = first_part + second_part
            negation_word = ", not a" if ele.tag_ in sigular_set else ", not"
            not_part = negation_word + \
                       sentence[start_index + len(start_element[0].text):end_index]
            if sentence[end_index + 1] != "," and sentence[end_index + 1] != ".":
                not_part = not_part + ","

            negation_sentence = first_part + not_part + second_part
            return original_sentence + "\t" + sentence + "\t" + negation_sentence
    return ""


def extact_sentences_with_adj_noun(sent, nlp):
    for ele in nlp(sent):
        if ele.pos_ == "ADJ" and ele.dep_ == "amod" and ele.head.pos_ == "NOUN":
            antonyms = get_antonyms(ele.text)
            real_antonyms = list(set((lemma.name() for lemma in antonyms if lemma.synset().pos() == 'a')))
            if real_antonyms:
                yield ((sent, ele.text, ele.head.text, "|".join(real_antonyms)))


def has_verb(nlp, sent):
    verb_list = [ele for ele in nlp(sent) if ele.dep_ == "nsubj"]
    return len(verb_list) > 0


def get_antonyms(word):
    antonym_list = [anto for syn in wordnet.synsets(word) for lemma in syn.lemmas() for anto in lemma.antonyms()]
    return antonym_list


def filter_noun_adj(snli_path, output_path):
    snli_list = read_file(snli_path,
                          preprocess=lambda x: x.strip().split("\t"))
    snli_list = filter(
        lambda x: not x[1].strip() in {"little", "small"} or not x[2].strip() in {"boy", "girl", "child", "children",
                                                                                  "baby"},
        snli_list)
    new_snli_list = filter(lambda x: not x[1] in {"green", "other", "musical"}, snli_list)
    new_quad_list = []
    antonym_dict = {"same": "different", "different": "same", "long": "short", "short": "long",
                    "older": "young", "younger": "old"}
    for quad in snli_list:
        if quad[1] == "same":
            quad[3] = "different"
        if quad[1] == "long":
            quad[3] = "short"
        if quad[1] == "short":
            quad[3] = "long"

        new_quad_list.append(quad)

    output_list_to_file(new_quad_list, output_path, process=lambda x: "\t".join(x))


def a_to_an(token_list):
    new_antonym_sent = []
    char_set = {"a", "e", "i", "o", "u"}
    for idx, word in enumerate(token_list):
        if idx == len(token_list) - 1 or not token_list[idx + 1]:
            new_antonym_sent.append(word)
            continue
        if word not in {"a", "A"} or token_list[idx + 1][0] not in char_set:
            new_antonym_sent.append(word)
            continue
        if word == "a" and token_list[idx + 1][0] in char_set:
            new_antonym_sent.append("an")
        if word == "A" and token_list[idx + 1][0] in char_set:
            new_antonym_sent.append("An")

    return " ".join(new_antonym_sent)


def generate_compositional_not_dataset(snli_list):
    snli_list = filter(lambda x: len(x[3].split("|")) == 1, snli_list)
    replace_dict = {"boy": "man", "girl": "woman", "boys": "men", "girls": "women"}
    person_set = {"man", "woman", "men", "women", "boy", "boys", "girl", "girls", "kid", "kids", "lady", "ladies",
                  "gentleman", "gentlemen", "male", "female"}
    result_list = []
    for quad in snli_list:
        sent = quad[0]
        if quad[1].strip().lower() in {"green", "musical"}:
            continue

        token_list = nlp(sent)
        for token in token_list:
            head_noun = token.head
            if token.text == quad[1] and head_noun.text == quad[2]:
                replace_template = "{0} who {1} {2}" if quad[2] in person_set else "{0} which {1} {2}"
                if quad[1] == "young" and quad[2] in replace_dict:
                    first = sent[:token.idx]
                    replace_word = quad[3] + " " + replace_dict[quad[2]]
                    end_index = head_noun.idx + len(head_noun.text)

                else:
                    first = sent[:token.idx]
                    replace_word = quad[3] + " "
                    end_index = head_noun.idx

                second = quad[0][end_index:]
                antonym_sent = first + replace_word + second
                new_antonym_sent = a_to_an(antonym_sent.split(" "))

                linking_verb = "is" if head_noun.tag_ in {"NN", "NNP"} else "are"
                syn_second_index = head_noun.idx + len(head_noun.text)
                synonym_sent = first + replace_template.format(quad[2], linking_verb, quad[1]) + quad[0][
                                                                                                 syn_second_index:]
                result_list.append([sent, synonym_sent, new_antonym_sent])


def is_sentence_with_clause(sentence, parser):
    for ele in parser(sentence):
        if ele.dep_ == "ccomp" and ele.head.text in {"say", "said", "says", "think", "state", "states", "stated",
                                                     "thought", "thinks",
                                                     "believe",
                                                     "believes", "believed", "claim",
                                                     "claims",
                                                     "claimed"} and ele.head.dep_ == "ROOT":
            return True

    return False


def generate_negation_sentence(msrp_iter):
    replace_dict = {"say": "deny", "says": "denies", "said": "denied", "think": "doubt", "thinks": "doubts",
                    "thought": "doubted", "believe": "suspect", "believes": "suspects", "believed": "suspected",
                    "claim": "disclaim", "claims": "disclaims", "claimed": "disclaimed"}
    for first, second in msrp_iter:
        for ele in nlp(first):
            opinion_verb = ele.head
            if ele.dep_ == "ccomp" and opinion_verb.text in replace_dict:
                replace_word = replace_dict[opinion_verb.text]
                end_index = ele.head.idx + len(opinion_verb.text)
                negation_sent = first[:opinion_verb.idx] + replace_word + first[end_index:]
                yield (first + "\t" + second + "\t" + negation_sent)


def filter_mrpc(file_path):
    msrp_iter = read_file(file_path, preprocess=lambda x: x.strip().split("\t"))
    msrp_iter = filter(lambda x: len(x) == 5 and x[0] == "1", msrp_iter)
    msrp_iter = map(lambda x: (x[3], x[4]), msrp_iter)
    for first, second in msrp_iter:
        if is_sentence_with_clause(first, nlp):
            yield (first + "\t" + second)
        elif is_sentence_with_clause(second, nlp):
            yield (second + "\t" + first)


def sentecnes_without_opinion_negation(sentence, parser):
    """

    :type sentence: str
    :type parser: spacy.lang.en.English
    :return:
    """
    for ele in parser(sentence):
        if ele.dep_ == "ccomp" and ele.head.text in {"say", "said", "says", "think", "state", "states", "stated",
                                                     "thought", "thinks",
                                                     "believe",
                                                     "believes", "believed", "claim",
                                                     "claims",
                                                     "claimed"} and ele.head.dep_ == "ROOT" and ele.head.idx > ele.idx:
            return True

    return False


def sentences_with_word_not_adj(sentence, parser):
    """

    :type sentence: str
    :type parser: spacy.lang.en.English
    :return:
    """
    for ele in parser(sentence):
        if ele.pos_ == "ADJ" and ele.dep_ == "amod" and ele.head.pos_ == "NOUN" and ele.idx > ele.head.idx:
            return False

    return True


def fix_bugs_in_sentence(sent, parser):
    for token in parser(sent):
        if token.dep_ == "relcl" and token.text == "is":
            if token.head.tag_ in {"NNS", "NNPS"}:
                return sent[:token.idx] + "are" + sent[token.idx + len(token.text):]
    return sent


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    msrp_dir = "/home/zxj/Downloads/new_corpus"
    file_path = os.path.join(msrp_dir, "filtered_opinion_negation_triplets.txt")
    negation_pattern = re.compile(r" 't")
    msrp_iter = read_file(file_path, preprocess=lambda x: x.strip().split("\t"))
    for ele in msrp_iter:
        first = ele[0]
        if negation_pattern.search(first):
            continue
        second = negate_word_msr(first, nlp)
        print(first + "\t" + ele[1] + "\t" + second)
