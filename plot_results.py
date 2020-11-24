import os
import json
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    data_dir = "/home/zxj/Data/experiment_results"
    file_name_list = ["adjective_compositionality", "argument_compositionality", "factual_relatedness",
                      "negation_detection", "argument_sensitivity", "negation_variant"]

    name_mapping = {"glove": "Glove Avg", "skip_thought": "Skip-Thought", "infersentV1": "InferSent V1", "infersentV2": "InferSent V1",
                    "elmo_average": "ELMO Average", "elmo_max": "ELMO Max"}

    prop = list(plt.rcParams['axes.prop_cycle'])
    prop.append({'color': '#F1C40F'})
    prop.append({'color': '#6E2C00'})
    prop = [ele['color'] for ele in prop]
    plt.tight_layout()
    avg_suffix = "{0}-AVG"
    cls_suffix = "{0}-CLS"
    bert_model_names = ["BERT-BASE", "BERT-LARGE", "SBERT-BASE", "SBERT-LARGE"]
    for idx, name in enumerate(file_name_list):
        file_path = os.path.join(data_dir, "{0}_result.txt".format(name))
        with open(file_path, encoding="utf-8") as file:
            result_dict = (json.load(file))
            for bert_name in bert_model_names:
                avg_key, cls_key = avg_suffix.format(bert_name), cls_suffix.format(bert_name)
                avg_acc = result_dict[avg_key]
                cls_acc = result_dict[cls_key]
                if avg_acc > cls_acc:
                    result_dict.pop(cls_key, None)
                else:
                    result_dict.pop(avg_key, None)

            key_list = []
            value_list = []
            for key, value in result_dict.items():
                if key in name_mapping:
                    key_list.append(name_mapping[key])
                else:
                    key_list.append(key)
                value_list.append(value)
            y_pos = np.arange(len(value_list))

            plt.barh(y_pos, value_list, color=prop)
            plt.yticks(y_pos, key_list)
            plt.xlabel('Accuracy')
            output_name = os.path.join(data_dir, name)
            fig = plt.gcf()
            fig.set_size_inches(16, 9)
            fig.savefig('/home/zxj/Data/experiment_results/{0}.png'.format(name), format='png', dpi=300)
            fig.clear()
