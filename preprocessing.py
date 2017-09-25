# encoding=utf8

import json
import logging
import random
import re
import sys
import string


prefix_re = re.compile("# text = ")
SENT_ID = "sent_id"
DOC_ID = "newdoc id"
PUNCT = "PUNCT"
FLOAT_NUM = re.compile("^[0-9]+\.")

def remove_rebundant(input_line):
    if "# text = " in input_line:
        return prefix_re.sub("", input_line)
    elif "# newdoc id = " in input_line or "# sent_id" in input_line:
        return "----"
    else:
        return input_line


def file_to_list(file_path, delimiter="->"):
    try:
        with open(file_path, encoding="utf8") as file:
            contents = file.read().split("\n\n")
            for sentence in contents:
                lines = sentence.split("\n")
                lines = [line for line in lines if not (SENT_ID in line or DOC_ID in line)]

                new_lines = [ele.split("\t") for ele in lines[1:]]
                first = [ele[1] for ele in new_lines]
                new_lines = [ele for ele in new_lines if not FLOAT_NUM.match(ele[0])
                             and ele[3] != PUNCT and ele[6] != "0" and ele[3] != "SYM"]
                first_str = prefix_re.sub("", lines[0])

                positive_pairs = [(int(ele[0]) - 1, int(ele[6]) - 1) for ele in new_lines]
                negative_pairs = [(ele[0], generate_negative_sample(first, ele)) for ele in positive_pairs if len(first) > 0]
                negative_pairs = [ele for ele in negative_pairs if ele[1] != -1]

                positive_pairs = [first[ele[0]] + delimiter + first[ele[1]] for ele in positive_pairs]
                negative_pairs = [first[ele[0]] + delimiter + first[ele[1]] for ele in negative_pairs]

                if new_lines:
                    out_dict = ({"sentence": first_str,
                                         "positive": ",".join(positive_pairs),
                                        "negative": ",".join(negative_pairs)})
                    print(json.dumps(out_dict))

    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)


def generate_negative_sample(string_array, positive_tuple):
    max_length = len(string_array)
    if max_length <= 1:
        return -1
    selection = [i for i in range(max_length) if i != positive_tuple[1] and string_array[i] not in string.punctuation]
    return random.choice(selection)


def output_list(input_list, output_path):
    try:
        with open(output_path, "w+", encoding="utf8") as file:
            for line in input_list:
                file.write(line + "\n")

    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)


if __name__ == '__main__':
    path_train = "/Users/zxj/Downloads/ud-treebanks-v2.0/UD_English/en-ud-train.conllu"
    path_test = "/Users/zxj/Downloads/sentence_evaluation/UD_English/en-ud-dev.conllu"
    file_to_list(path_test)
