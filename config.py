import argparse


def init_argument_parser():
    parser = argparse.ArgumentParser(description="Sentence Evaluation")
    parser.add_argument("--glove-path", type=str, metavar="N",
                        default="/home/zxj/Downloads/data/glove.840B.300d.txt",
                        help="path of glove file")

    parser.add_argument("--infer-sent-model-path", type=str, metavar="N",
                        default="/home/zxj/Downloads/data/infersent.allnli.pickle",
                        help="path of InferSent models")

    parser.add_argument("--sent2vec-model-path", type=str, metavar="N",
                        default="/mnt/wiki_unigrams.bin",
                        help="path of sent2vec model")
    parser.add_argument("--data-path", type=str,
                        default="/home/zxj/Downloads/new_corpus",
                        help="path of data file")
    return parser
