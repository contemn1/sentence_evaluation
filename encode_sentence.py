import json
import logging
import os
import sys
from functools import reduce

import numpy as np
import sent2vec
import spacy
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from nltk.tokenize import word_tokenize
from pytorch_pretrained_bert import BertModel, BertTokenizer
from scipy import logical_and
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

from IOUtil import get_glove, restore_skipthought
from IOUtil import get_word_dict, read_file
from config import init_argument_parser
from dataset.custom_dataset import TextIndexDataset
from models import InferSent


def load_sentences(file_path):
    try:
        with open(file_path, encoding="utf8") as file:
            sentence_dict = [json.loads(line) for line in file]
            return sentence_dict
    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)


def resume_model(model_path, dict_path, version, use_cuda):
    device = torch.device('cuda' if use_cuda else 'cpu')
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': version}
    infer_sent = InferSent(params_model)
    infer_sent.load_state_dict(torch.load(model_path, map_location=device))

    infer_sent.set_w2v_path(dict_path)
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
    sent_list = [ele.strip() for arr in sent_list for ele in arr]
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
        model = resume_model(model_path, word2vec_path, version, use_cuda)  # InferSent
        model.build_vocab(sentences, tokenize=True)
        if use_cuda:
            model = model.cuda()

        return model.encode(sentences, bsize=batch_size, tokenize=True,
                            verbose=False)

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


def output_results(embeddings, calculate_accuracy=normal_accuracy, verbose=False, output_path="average_scores_new.txt"):
    results = calculate_pairwise_similarity(embeddings)
    score_array = np.mean(axis=0, a=results)
    if verbose:
        output = 100 * results
        np.savetxt(output_path, output, fmt='%10.3f', delimiter='\t')

    accuracy = calculate_accuracy(results)

    result_arr = score_array.tolist()
    result_arr.append(accuracy)
    return accuracy


def get_results(file_paths, config):
    for path in file_paths:
        triples = sentences_unfold(path, delimiter="\t")
        glove_result = output_results(get_embedding_from_glove(config.glove_path, power=1)(triples),
                                      calculate_accuracy=normal_accuracy)
        print("Glove Avg \t &　{0} \t \\\\ ".format("\t &".join(glove_result)))

        p_means_result = output_results(get_embedding_from_glove(config.glove_path, power=3)(triples),
                                        calculate_accuracy=normal_accuracy)
        print("P Means \t &　{0} \t \\\\ ".format("\t & ".join(p_means_result)))

        sent2vec_result = output_results(get_embedding_from_sent2vec(config.sent2vec_model_path)(triples),
                                         calculate_accuracy=normal_accuracy, verbose=False,
                                         output_path="set2vec_clause_relatedness")
        print(
            "Sent2Vec \t &　{0} \t \\\\ ".format("\t & ".join(sent2vec_result)))

        infer_sent_result = output_results(get_embedding_from_infersent(config.model_path)(triples),
                                           calculate_accuracy=normal_accuracy, verbose=True,
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
        data_loader = DataLoader(dataset, batch_size=64, num_workers=0,
                                 collate_fn=dataset.collate_fn_one2one)
        pool_result = None
        bert = model
        average_pooling_result = None
        max_pooling_result = None

        for ids, masks in data_loader:
            if torch.cuda.is_available():
                ids = ids.cuda()
                masks = masks.cuda()
                bert = model.cuda()

            with torch.no_grad():
                encoded_layers, pooled_output = bert(ids, attention_mask=masks,
                                                     output_all_encoded_layers=False)
                max_pooling_batch, _ = torch.max(encoded_layers, dim=1)
                average_pooling_batch = get_average_pooling(encoded_layers, masks)

                if pool_result is None:
                    pool_result = pooled_output.cpu().numpy()
                else:
                    pool_result = np.vstack((pool_result, pooled_output.cpu().numpy()))

                if max_pooling_result is None:
                    max_pooling_result = max_pooling_batch.cpu().numpy()
                else:
                    max_pooling_result = np.vstack((max_pooling_result, max_pooling_batch.cpu().numpy()))

                if average_pooling_result is None:
                    average_pooling_result = average_pooling_batch.cpu().numpy()
                else:
                    average_pooling_result = np.vstack((average_pooling_result, average_pooling_batch.cpu().numpy()))

        return pool_result, average_pooling_result, max_pooling_result

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
    sum_embeddings = torch.bmm(masks.unsqueeze(1).float(), embeddings).squeeze(1)
    average_embeddings = sum_embeddings / masks.sum(dim=1, keepdim=True).float()
    return average_embeddings


def get_max_pooling(embeddings):
    max_pooling_embeddings, _ = torch.max(embeddings, 1)
    return max_pooling_embeddings


def concatenation_encode(data_path):
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
        max_pooling_result = output_results(max_pooling_tensor.cpu().numpy(), calculate_accuracy=accuracy_function[idx])
        average_bert_result = output_results(average_bert_tensor.cpu().numpy(),
                                             calculate_accuracy=accuracy_function[idx])
        max_bert_result = output_results(max_bert_tensor.cpu().numpy(), calculate_accuracy=accuracy_function[idx])

        print("Result of average pooling bert on {0} dataset is: --------".format(file_name_list[idx]))
        print("\t& ".join(average_bert_result) + """\\""")
        print("Result of max pooling bert on {0} dataset is: --------".format(file_name_list[idx]))
        print("\t& ".join(max_bert_result) + """\\""")

        print("Result of average pooling  concatenation on {0} dataset is: --------".format(file_name_list[idx]))
        print("\t& ".join(average_pooling_result) + """\\""")
        print("Result of max pooling concatenation on {0} dataset is: --------".format(file_name_list[idx]))
        print("\t& ".join(max_pooling_result) + """\\""")


def calculate_sbert_accuracy(sentence_list, accuracy_func):
    new_name_dict = {"SBERT-BASE-AVG": "bert-base-nli-mean-tokens",
                     "SBERT-LARGE-AVG": "bert-large-nli-mean-tokens",
                     "SBERT-BASE-CLS": "bert-base-nli-cls-token",
                     "SBERT-LARGE-CLS": "bert-large-nli-cls-token",
                     "SRoBERTa-BASE-AVG": "roberta-base-nli-mean-tokens",
                     "SRoBERTa-LARGE-AVG": "roberta-large-nli-mean-tokens"}
    result_dict = {}
    for output_name, model_name in new_name_dict.items():
        model = SentenceTransformer(model_name)
        embedding_list = model.encode(sentence_list, batch_size=32)
        embeddings = np.vstack(embedding_list)
        accuracy_result = output_results(embeddings=embeddings,  calculate_accuracy=accuracy_func)
        result_dict[output_name] = accuracy_result
    return result_dict


def main():
    config = init_argument_parser().parse_args()
    file_name_list = ["adjective_compositionality", "argument_compositionality", "factual_relatedness",
                      "negation_detection", "argument_sensitivity", "negation_variant"]
    accuracy_calculation_methods = [normal_accuracy, normal_accuracy, normal_accuracy,
                                    normal_accuracy, normal_accuracy, negation_variant_accuracy]
    infer_sent_dict_path = [config.glove_path, config.fast_text_path]

    bert_base_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_large_tokenzier = BertTokenizer.from_pretrained("bert-large-uncased")

    # bert_base_snli = BertModel.from_pretrained('bert-base-uncased', state_dict=model_state_dict)

    bert_base_model = BertModel.from_pretrained("bert-base-uncased")
    bert_large_model = BertModel.from_pretrained("bert-large-uncased")
    bert_base_model.eval()
    bert_large_model.eval()

    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    elmo_model = Elmo(options_file, weight_file, 1, dropout=0)

    glove_encoder = get_embedding_from_glove(config.glove_path)


    for idx, name in enumerate(file_name_list):
        file_path = os.path.join(config.data_path, "{0}.txt".format(name))
        triplets = sentences_unfold(file_path=file_path, delimiter="\t")
        nlp = spacy.load('en_core_web_sm')

        tokenized_triplets = [[token.text for token in nlp(sent)] for sent in triplets]

        result_dict = {"glove": 0, "infersentV1": 0, "infersentV2": 0, "skip_thought": 0}

        accuracy_calculation_func = accuracy_calculation_methods[idx]

        result_dict["glove"] = output_results(glove_encoder(triplets), accuracy_calculation_func)

        for version in range(1, 3):
            model_path = config.infer_sent_model_path.format(version)
            infer_sent_encoder = get_embedding_from_infersent(model_path, infer_sent_dict_path[version - 1],
                                                              use_cuda=True)
            dict_key = "infersentV{0}".format(version)
            result_dict[dict_key] = output_results(infer_sent_encoder(triplets), accuracy_calculation_func)
        
        skip_thought_encoder = restore_skipthought(config.skipthought_path, config.skipthought_model_name,
                                            config.skipthought_embeddings, config.skipthought_vocab_name)

        skip_thought_embeddings = skip_thought_encoder.encode(triplets, batch_size=64, use_norm=False)
        result_dict["skip_thought"] = output_results(skip_thought_embeddings, accuracy_calculation_func)

        base_encoder = get_embedding_from_bert(bert_base_model, bert_base_tokenizer)

        cls_pooling_base, average_pooling_base, max_pooling_base = base_encoder(triplets)

        result_dict["BERT-BASE-CLS"] = output_results(cls_pooling_base, accuracy_calculation_func)
        result_dict["BERT-BASE-AVG"] = output_results(average_pooling_base, accuracy_calculation_func)

        large_encoder = get_embedding_from_bert(bert_large_model, bert_large_tokenzier)
        cls_pooling_large, average_pooling_large, max_pooling_large = large_encoder(triplets)
        result_dict["BERT-LARGE-CLS"] = output_results(cls_pooling_large, accuracy_calculation_func)
        result_dict["BERT-LARGE-AVG"] = output_results(average_pooling_large, accuracy_calculation_func)

        elmo_average, elmo_max = elmo_embeddings(tokenized_triplets, model=elmo_model)
        result_dict["elmo_average"] = output_results(elmo_average, accuracy_calculation_func)
        result_dict["elmo_max"] = output_results(elmo_max, accuracy_calculation_func)

        new_result = calculate_sbert_accuracy(triplets, accuracy_func=accuracy_calculation_func)
        for key, value in new_result.items():
            result_dict[key] = value
        output_path = "/home/zxj/Data/experiment_results/{0}_result.txt".format(name)
        with open(output_path, mode="w+") as file:
            json.dump(result_dict, file)

def get_skip_thought_accuracy():
    config = init_argument_parser().parse_args()
    file_name_list = ["negation_detection", "argument_sensitivity", "negation_variant"]
    accuracy_calculation_methods = [normal_accuracy, normal_accuracy, negation_variant_accuracy]

    skip_thought_encoder = restore_skipthought(config.skipthought_path, config.skipthought_model_name,
                                               config.skipthought_embeddings, config.skipthought_vocab_name)

    for idx, name in enumerate(file_name_list):
        file_path = os.path.join(config.data_path, "{0}.txt".format(name))
        triplets = sentences_unfold(file_path=file_path, delimiter="\t")
        output_path = "/home/zxj/Data/experiment_results/{0}_result.txt".format(name)
        with open(output_path, mode='r+') as json_file:
            json_dict = json.load(json_file)
            embeddings = skip_thought_encoder.encode(triplets, batch_size=64, use_norm=False)
            accuracy_calculation_func = accuracy_calculation_methods[idx]
            json_dict['skip_thought'] = output_results(embeddings, calculate_accuracy=accuracy_calculation_func)
            json_file.write("\n")
            json_file.write(json.dumps(json_dict))


if __name__ == '__main__':
    main()
