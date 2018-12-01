from models import InferSent
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
from scipy import logical_and
import sent2vec
from config import init_argument_parser
import os
from dataset.custom_dataset import TextIndexDataset
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertModel
from allennlp.modules.elmo import Elmo, batch_to_ids


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


def resume_model(model_path, glove_path, version, use_cuda):
    device = torch.device('cuda' if use_cuda else 'cpu')
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': version}
    infer_sent = InferSent(params_model)
    infer_sent.load_state_dict(torch.load(model_path, map_location=device))
    infer_sent.set_w2v_path(glove_path)
    return infer_sent


def calculate_pairwise_similarity(embd, group_size=3):
    results = []
    embd = embd.reshape((-1, group_size, embd.shape[1]))
    for ele in embd:
        res_mat = cosine_similarity(ele)
        results.append(
            [res_mat[0][1].item(), res_mat[0][2].item(), res_mat[1][2].item()])

    return np.array(results)


def sentences_unfold(file_path, delimiter="\001",
                     predicate=lambda x: len(x) == 3):
    sent_list = read_file(file_path,
                          preprocess=lambda ele: ele.split(delimiter))
    sent_list = (arr for arr in sent_list if predicate(arr))
    sent_list = [ele for arr in sent_list for ele in arr]
    return sent_list


def load_sick(sick_path="/Users/zxj/Downloads/SICK/SICK.txt"):
    file_list = read_file(sick_path)
    file_list = (ele.split("\t")[1:7] for ele in file_list if
                 not ele.startswith("pair_ID"))
    file_list = ([ele[0], ele[1], ele[2]] for ele in file_list)
    return file_list


def power_mean(p):
    def calculate_mean(array):
        if p == 1:
            return np.mean(array, axis=0)

        mean = np.power(array, p).mean(axis=0).astype('complex')
        return np.power(mean, (1.0 / p)).real

    return calculate_mean


def get_embedding_from_glove(glove_path, power=1):
    def get_embedding(sentences):
        word_dict = get_word_dict(sentences)
        glove_dict = get_glove(glove_path, word_dict)
        sentences = [word_tokenize(sent) for sent in sentences]
        sentences = [[glove_dict[word] for word in sent if word in glove_dict]
                     for sent in sentences]
        result_list = []
        print(len(sentences))
        for idx, vec in enumerate(sentences):
            result = power_mean(1)(vec)
            result_list.append(result)

        average = np.array(result_list, dtype=np.float32)
        if power > 1:
            for index in range(2, power + 1):
                average = np.concatenate(
                    (average, [power_mean(index)(vec) for vec in sentences]),
                    axis=1)

        return average

    return get_embedding


def get_embedding_from_infersent(model_path, word2vec_path, batch_size=128,
                                 version=2, use_cuda=True):
    def get_infersent_embedding(sentences):
        model = resume_model(model_path, word2vec_path, version, use_cuda)
        model.build_vocab(sentences, tokenize=True)
        return model.encode(sentences, bsize=batch_size, tokenize=True,
                            verbose=True)

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
    sick_list = load_sick()
    language_model = spacy.load('en')
    filtered_sick_list = filter(
        lambda arr: sent_no_clause(language_model, arr[0])
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
    return [snli_dict["sentence1"], snli_dict["sentence2"],
            snli_dict["gold_label"]]


def encode_triples(file_path, delimiter, triple_to_embedding):
    triplets = sentences_unfold(file_path=file_path, delimiter=delimiter)
    triplets = [ele.strip() for ele in triplets]
    embeddings = triple_to_embedding(triplets)
    return embeddings


def normal_accuracy(results):
    true_results = results[:, 0] > results[:, 1]
    accuracy = np.sum(true_results) / len(true_results)
    return accuracy


def negation_variant_accuracy(results):
    true_results1 = results[:, 0] < results[:, 2]
    true_results2 = results[:, 1] < results[:, 2]
    true_result = logical_and(true_results1, true_results2)
    accuracy = np.sum(true_result) / len(true_result)
    return accuracy


def output_results(embeddings, verbose=False,
                   calculate_accuracy=normal_accuracy,
                   output_path="average_scores"):
    results = calculate_pairwise_similarity(embeddings)
    score_array = np.mean(axis=0, a=results)
    if verbose:
        output = 100 * results
        np.savetxt(output_path, output, fmt='%10.3f', delimiter='\t')

    accuracy = calculate_accuracy(results)

    result_arr = score_array.tolist()
    result_arr.append(accuracy)
    result_arr = ["{:.2f}\\%".format(ele * 100) for ele in result_arr]
    return result_arr


def get_results(file_paths):
    for path in file_paths:
        triples = sentences_unfold(path, delimiter="\t")
        glove_result = output_results(
            get_embedding_from_glove(glove_path, power=1)(triples),
            calculate_accuracy=normal_accuracy)
        print("Glove Avg \t &　{0} \t \\\\ ".format("\t &".join(glove_result)))

        p_means_result = output_results(
            get_embedding_from_glove(glove_path, power=3)(triples),
            calculate_accuracy=normal_accuracy)
        print("P Means \t &　{0} \t \\\\ ".format("\t & ".join(p_means_result)))

        sent2vec_result = output_results(
            get_embedding_from_sent2vec(sent2vec_model_path)(triples),
            calculate_accuracy=normal_accuracy,
            verbose=False,
            output_path="set2vec_clause_relatedness")
        print(
            "Sent2Vec \t &　{0} \t \\\\ ".format("\t & ".join(sent2vec_result)))

        infer_sent_result = output_results(
            get_embedding_from_infersent(model_path)(triples),
            calculate_accuracy=normal_accuracy,
            verbose=True,
            output_path="infersent_clause_relatedness")

        print("Infersent \t &　{0} \t   \\\\ \\bottomrule".format(
            "\t & ".join(infer_sent_result)))


def get_embedding_from_bert(model, tokenizer):
    """
    :type model: BertModel
    :type tokenizer: BertTokenizer
    :return:
    """

    def bert_embeddings(sentences):
        dataset = TextIndexDataset(sentences, tokenizer, True)
        data_loader = DataLoader(dataset, batch_size=72, num_workers=0,
                                 collate_fn=dataset.collate_fn_one2one)
        result = []
        for ids, masks, _ in data_loader:
            if torch.cuda.is_available():
                ids = ids.cuda()
                masks = masks.cuda()
            encoded_layers, _ = model(ids, attention_mask=masks,
                                      output_all_encoded_layers=False)
            average_embeddings = torch.mean(encoded_layers,
                                            dim=1)  # torch.Tensor
            result.append(average_embeddings.detach().cpu())
        result = np.vstack(result)
        return result

    return bert_embeddings


def elmo_embeddings(sentences, model):
    model.eval()
    batch_size = 48
    average_pooling_tensor = None
    max_pooling_tensor = None
    for idx in range(0, len(sentences), batch_size):
        id_batch = batch_to_ids(sentences[idx: idx + batch_size])
        if torch.cuda.is_available():
            model = model.cuda()
            id_batch = id_batch.cuda()
        with torch.no_grad():
            embeddings = model(id_batch)
            elmo_representations = embeddings["elmo_representations"][0]  # type: torch.Tensor
            masks = (embeddings["mask"]).float()  # type: torch.Tensor
            average_embeddings = get_average_pooling(elmo_representations, masks)
            max_pooling_embeddings = get_max_pooling(elmo_representations)
            average_pooling_tensor = average_embeddings if average_pooling_tensor is None else torch.cat(
                [average_pooling_tensor, average_embeddings], dim=0)
            max_pooling_tensor = max_pooling_embeddings if max_pooling_tensor is None else torch.cat(
                [max_pooling_tensor, max_pooling_embeddings], dim=0)
    return average_pooling_tensor.detach().cpu().numpy(), max_pooling_tensor.detach().cpu().numpy()


def get_average_pooling(embeddings, masks):
    sum_embeddings = torch.bmm(masks.unsqueeze(1), embeddings).squeeze(1)
    average_embeddings = sum_embeddings / masks.sum(dim=1, keepdim=True)
    return average_embeddings


def get_max_pooling(embeddings):
    max_pooling_embeddings, _ = torch.max(embeddings, 1)
    return max_pooling_embeddings


if __name__ == '__main__':
    parser = init_argument_parser()
    config = parser.parse_args()
    glove_path = config.glove_path
    model_path = config.infer_sent_model_path
    sent2vec_model_path = config.sent2vec_model_path
    data_path = config.data_path
    file_name_list = ["negation_detection.txt", "negation_variant.txt",
                      "clause_relatedness.txt", "argument_sensitivity.txt",
                      "fixed_point_inversion.txt"]
    accuracy_function = [normal_accuracy, negation_variant_accuracy,
                         normal_accuracy, normal_accuracy, normal_accuracy]
    file_path_list = [os.path.join(data_path, ele) for ele in file_name_list]
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    elmo_model = Elmo(options_file, weight_file, 1)
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    elmo_model.eval()
    bert_model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for idx, path in enumerate(file_path_list):
        average_pooling_tensor = None
        max_pooling_tensor = None
        average_bert_tensor = None
        max_bert_tensor = None
        sentences = sentences_unfold(path, delimiter="\t")
        dataset = TextIndexDataset(sentences, tokenizer, True)
        data_loader = DataLoader(dataset, batch_size=48, num_workers=0,
                                 collate_fn=dataset.collate_fn_one2one)
        for ids, masks, elmo_ids in data_loader:
            masks = masks.float()
            if torch.cuda.is_available():
                ids = ids.cuda()
                masks = masks.cuda()
                elmo_ids = elmo_ids.cuda()
                bert_model = bert_model.cuda()
                elmo_model = elmo_model.cuda()
            with torch.no_grad():
                encoded_bert_layers, _ = bert_model(ids, attention_mask=masks,
                                                    output_all_encoded_layers=False)
                elmo_dict = elmo_model(elmo_ids)
                elmo_representations = elmo_dict["elmo_representations"][0]  # type: torch.Tensor
                elmo_mask = elmo_dict["mask"]
                elmo_mask = elmo_mask.float()
                concatenated_layers = torch.cat((encoded_bert_layers, elmo_representations), dim=2)

                average_elmo = get_average_pooling(elmo_representations, elmo_mask)
                max_elmo = get_max_pooling(elmo_representations)

                average_embeddings = get_average_pooling(concatenated_layers, masks)
                max_pooling_embeddings = get_max_pooling(concatenated_layers)
                average_pooling_tensor = average_embeddings if average_pooling_tensor is None else torch.cat(
                    [average_pooling_tensor, average_embeddings], dim=0)
                max_pooling_tensor = max_pooling_embeddings if max_pooling_tensor is None else torch.cat(
                    [max_pooling_tensor, max_pooling_embeddings], dim=0)
                average_bert_tensor = average_elmo if average_bert_tensor is None else torch.cat(
                    [average_bert_tensor, average_elmo], dim=0)
                max_bert_tensor = max_elmo if max_bert_tensor is None else torch.cat(
                    [max_bert_tensor, max_elmo], dim=0)

        average_pooling_result = output_results(average_pooling_tensor.cpu().numpy(),
                                                calculate_accuracy=accuracy_function[idx])
        max_pooling_result = output_results(max_pooling_tensor.cpu().numpy(),
                                            calculate_accuracy=accuracy_function[idx])
        average_bert_result = output_results(average_bert_tensor.cpu().numpy(),
                                             calculate_accuracy=accuracy_function[idx])
        max_bert_result = output_results(max_bert_tensor.cpu().numpy(),
                                         calculate_accuracy=accuracy_function[idx])

        print("Result of average pooling bert on {0} dataset is: --------".format(file_name_list[idx]))
        print("\t& ".join(average_bert_result) + """\\""")
        print("Result of max pooling bert on {0} dataset is: --------".format(file_name_list[idx]))
        print("\t& ".join(max_bert_result) + """\\""")

        print("Result of average pooling  concatenation on {0} dataset is: --------".format(file_name_list[idx]))
        print("\t& ".join(average_pooling_result) + """\\""")
        print("Result of max pooling concatenation on {0} dataset is: --------".format(file_name_list[idx]))
        print("\t& ".join(max_pooling_result) + """\\""")
