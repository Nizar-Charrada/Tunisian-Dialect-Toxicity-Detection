import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import re


class TransData(Dataset):

    """Dataset class for transliteration
    Args:
        data (pd.DataFrame): dataframe containing source and target language
        source (str): source language
        source_vocab (Lang): source language vocabulary
        target (str): target language
        target_vocab (Lang): target language vocabulary
    """

    def __init__(self, data, source, source_vocab, target, target_vocab):
        self.data = data.copy()
        self.source_data = self.data[source].apply(source_vocab.tokenize)
        self.target_data = self.data[target].apply(target_vocab.tokenize)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input = torch.tensor(self.source_data.iloc[index], dtype=torch.long)
        target = torch.tensor(self.target_data.iloc[index], dtype=torch.long)
        return input, target


def custom_collate(batch):
    """Custom collate function to pad the batch"""
    inputs, outputs = zip(*batch)
    inputs = pad_sequence(inputs, padding_value=0)
    outputs = pad_sequence(outputs, padding_value=0)
    return inputs, outputs

