import numpy as np
from torch.utils.data import Dataset
import torch



class TextIndexDataset(Dataset):
    def __init__(self, word_sequence, tokenizer,
                 use_cuda=False):
        self.raw_texts = word_sequence
        self.tokenizer = tokenizer
        self.use_cuda = use_cuda
    def __len__(self):
        return len(self.raw_texts)

    def __getitem__(self, index):
        tokens = self.tokenizer.tokenize(self.raw_texts[index])
        new_tokens = ["[CLS]"] + tokens
        new_tokens.append("[SEP]")
        input_ids = self.tokenizer.convert_tokens_to_ids(new_tokens)
        return input_ids

    def collate_fn_one2one(self, batch_ids):
        '''
        Puts each data field into a tensor with outer dimension batch size"
        '''
        sequence_lengths = np.array([len(ele) for ele in batch_ids])
        padded_batch_ids = pad(batch_ids, sequence_lengths,
                               0)  # type: torch.Tensor
        print(padded_batch_ids.dtype)
        input_masks = padded_batch_ids > 0
        return padded_batch_ids, input_masks


def pad(sequence_raw, sequence_length, pad_id):
    def pad_per_line(index_list, max_length):
        return np.concatenate(
            (index_list, [pad_id] * (max_length - len(index_list))))

    max_seq_length = np.max(sequence_length)
    padded_sequence = np.array(
        [pad_per_line(x_, max_seq_length) for x_ in sequence_raw],
        dtype=np.int64)

    padded_sequence = torch.from_numpy(padded_sequence)

    return padded_sequence

