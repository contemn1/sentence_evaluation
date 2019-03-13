import os
import json
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    data_dir = "/home/zxj/Data/experiment_results"
    file_name_list = ["negation_detection", "negation_variant" , "argument_sensitivity",  "adjective_compositionality",
                      "factual_relatedness", "argument_compositionality"]

    row_list = ["glove", "skip_thought", "infersentV1", "infersentV2", "bert_base_cls", "bert_base_average",
                "bert_base_max", "bert_large_cls", "bert_large_average", "bert_large_max", "elmo_average", "elmo_max"]

    row_list = row_list[::-1]
    row_names = ["Glove Avg", "Skip-Thought", "InferSent V1", "InferSent V2", "BERT Base CLS", "BERT Base Avg",
                 "BERT Base Max", "BERT Large CLS", "BERT Large AVG", "BERT Large Max", "ELMO Average", "ELMO Max"]
    row_names = row_names[::-1]
    y_pos = np.arange(len(row_list))
    plt.tight_layout()

    for idx, name in enumerate(file_name_list):
        file_path = os.path.join(data_dir, "{0}_result.txt".format(name))
        with open(file_path, encoding="utf-8") as file:
            result_dict = (json.load(file))

            value_list = [result_dict[name] for name in row_list]
            plt.barh(y_pos, value_list, align='center', alpha=0.5)
            plt.yticks(y_pos, row_names)
            plt.xlabel('Accuracy')
            output_name = os.path.join(data_dir, name)
            fig = plt.gcf()
            fig.set_size_inches(10, 5)
            fig.savefig('/home/zxj/Data/experiment_results/{0}.png'.format(name), format='png', dpi=300)
            fig.clear()
