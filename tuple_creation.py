from encode_sentence import read_file
import re


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


if __name__ == '__main__':
    file_path = "/Users/zxj/PycharmProjects/sentence_evaluation/dataset/negative_no_unique.txt"
    file_generator = read_file(file_path, preprocess=lambda x: x.split("\t"))
    file_generator = ([tup[0], tup[1].strip(), negate_verb(tup[0])] for tup in file_generator)
    for ele in file_generator:
        print("\001".join(ele).strip())