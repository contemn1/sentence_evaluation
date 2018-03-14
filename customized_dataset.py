import numpy as np
import torch
from torch.utils.data import Dataset
from IOUtil import get_glove_dict

class TextDataset(Dataset):
    def __init__(self, glove_path, text_reader=None, text_data=None):
        if not text_reader and not text_data:
            raise ValueError("At least one of the text path and text data should not be not empty")

        if text_reader and text_data:
            raise ValueError("text path is mutually exclusive with text data")

        self.glove_dict = get_glove_dict(glove_path)
        if text_reader:
            self.data_x = text_reader.read_text()
        else:
            self.data_x = text_data

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        sentence = self.data_x[index]
        sentence = sentence.split(" ")
        gloves = [self.glove_dict[word] for word in sentence if word in self.glove_dict]
        glove_array = np.array(gloves)
        return torch.FloatTensor(glove_array)


class TextIndexDataset(Dataset):
    def __init__(self, word_to_index, text_reader=None, text_data=None):
        if not text_reader and not text_data:
            raise ValueError("At least one of the text path and text data should not be not empty")

        if text_reader and text_data:
            raise ValueError("text path is mutually exclusive with text data")

        self.word_to_index = word_to_index
        if text_reader:
            self.data_x = text_reader.read_text()
        else:
            self.data_x = text_data

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        sentence = self.data_x[index]
        sentence = sentence.split(" ")
        indices = [self.word_to_index[word] for word in sentence if word in self.word_to_index]
        indices_array = np.array(indices, dtype=np.int)
        return torch.LongTensor(indices_array)


class EmbeddingDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.embeddings = data_x
        self.labels = data_y

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, item):
        return torch.from_numpy(self.embeddings[item]), self.labels[item]
