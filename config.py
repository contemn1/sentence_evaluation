import argparse


def init_argument_parser():
    parser = argparse.ArgumentParser(description="Sentence Evaluation")
    parser.add_argument("--glove-path", type=str, metavar="N",
                        default="/home/zxj/Data/sent_embedding_data/glove.840B.300d.txt",
                        help="path of glove file")

    parser.add_argument("--infer-sent-model-path", type=str, metavar="N",
                        default="/home/zxj/Data/sent_embedding_data/infersent/infersent{0}.pkl",
                        help="path of InferSent models")

    parser.add_argument("--sent2vec-model-path", type=str, metavar="N",
                        default="/mnt/wiki_unigrams.bin",
                        help="path of sent2vec model")
    parser.add_argument("--fast-text-path", type=str,
                        default="/home/zxj/Data/sent_embedding_data/infersent/crawl-300d-2M.vec")
    parser.add_argument("--data-path", type=str,
                        default="/home/zxj/Downloads/new_corpus",
                        help="path of data file")

    parser.add_argument("--skipthought-path", type=str, metavar="N",
                        default="/home/zxj/Data/sent_embedding_data/skip_thoughts_uni_2017_02_02",
                        help="path of pre-trained skip-thought vectors model")

    parser.add_argument("--skipthought-model-name", type=str,
                        default="model.ckpt-501424",
                        help="name of pre-trained skip-thought vectors model")

    parser.add_argument("--skipthought-embeddings", type=str,
                        default="embeddings.npy",
                        help="name of pre-trained skip-thought word embeddings model")

    parser.add_argument("--skipthought-vocab-name", type=str,
                        default="vocab.txt",
                        help="name of pre-trained skip-thought vectors vocabulary")

    return parser
