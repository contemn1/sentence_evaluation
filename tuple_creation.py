from encode_sentence import read_file
import re
import spacy
from encode_sentence import read_file
import random

NO_PATTERN = re.compile("No")
CURRENT_PATTERN = re.compile("is|are|am")


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
    sents = read_file("test2.txt", preprocess=lambda x: extrac_verb_phase(language_model, x))
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


if __name__ == '__main__':
    person_names = list(read_file("/Users/zxj/Desktop/names_no_dup.txt",
                             lambda x: (x[0] + x[1:].lower()).strip()))

    verbs = list(read_file("/Users/zxj/Desktop/verbs", lambda x: x.strip()))
    first = verbs[:4]
    middle = verbs[4:10]
    verb_phases = list(read_file("/Users/zxj/PycharmProjects/sentence_evaluation/verb_phases_select.txt",
                            lambda x: x.strip()))

    complicate_triple(person_names, middle, verb_phases, 30)
