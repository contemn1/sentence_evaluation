from torch.utils.data import Dataset
from word_dict_test import read_file
from word_dict_test import split_string
import numpy as np
import json
from word_dict_test import get_glove_array
import torch
from IOUtil import unfold_domain


class TextDataset(Dataset):
    def __init__(self, glove_path, embedding_path, text_path=None, text_data=()):
        if not text_path and not text_data:
            raise ValueError("At least one of the text path and text data should not be not empty")

        if text_path and text_data:
            raise ValueError("text path is mutually exclusive with text data")

        glove_list = read_file(glove_path, split_string)
        self.glove_dict = {ele[0]: np.fromstring(ele[1], sep=" ") for ele in glove_list}
        if text_path:
            self.data_x, self.data_y = unfold_domain(read_file(text_path, lambda x: json.loads(x)))
        else:
            self.data_x, self.data_y = text_data

        self.sentence_embeddings = np.load(embedding_path)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        index, words = self.data_x[index]
        sentence_embedding = self.sentence_embeddings[index]
        word_embeddings = np.hstack((get_glove_array(words[0], self.glove_dict),
                                     get_glove_array(words[1], self.glove_dict)))
        all_embeddings = np.hstack((sentence_embedding, word_embeddings))
        x_tensor = torch.FloatTensor(all_embeddings)
        return x_tensor, self.data_y[index]


