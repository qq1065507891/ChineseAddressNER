import torch
from torch.utils.data import Dataset

from src.utils.config import config


class CustomDataset(Dataset):
    """
    数据集的类
    """
    def __init__(self, data):
        if config.train:
            self.input_ids, self.token_type_ids, self.attenion_mask, self.seq_len, self.labels = data
        else:
            self.input_ids, self.token_type_ids, self.attenion_mask, self.seq_len = data

    def __getitem__(self, item):
        if config.train:
            sample = [self.input_ids[item], self.token_type_ids[item],
                      self.attenion_mask[item], self.seq_len[item], self.labels[item]]
        else:
            sample = [self.input_ids[item], self.token_type_ids[item], self.attenion_mask[item], self.seq_len[item]]
        return sample

    def __len__(self):
        return len(self.input_ids)


def collate_fn(batch):
    if config.train:
        input_ids = [x[0] for x in batch]
        token_type_ids = [x[1] for x in batch]
        attention_masks = [x[2] for x in batch]
        seq_len = [x[3] for x in batch]
        labels = [x[4] for x in batch]
        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_masks),\
               torch.tensor(seq_len), torch.tensor(labels)
    else:
        input_ids = [x[0] for x in batch]
        token_type_ids = [x[1] for x in batch]
        attention_masks = [x[2] for x in batch]
        seq_len = [x[3] for x in batch]

        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_masks), torch.tensor(seq_len)


