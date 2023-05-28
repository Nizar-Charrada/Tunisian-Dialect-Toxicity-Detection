from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataloader.sampler import SmartBatchingSampler
from dataloader.collate import SmartBatchingCollate
import warnings
import argparse

warnings.filterwarnings("ignore")


class SmartBatchingDataset(Dataset):
    def __init__(self, df, tokenizer):
        super(SmartBatchingDataset, self).__init__()

        # Tokenize and convert each text input in the dataframe to a list of token IDs
        self._data = (
            (f"{tokenizer.cls_token} " + df["text"] + f" {tokenizer.sep_token}")
            .apply(tokenizer.tokenize)
            .apply(tokenizer.convert_tokens_to_ids)
            .to_list()
        )

        # If the dataframe contains labels, store them as a separate list
        self._targets = None
        if "label" in df.columns:
            self._targets = df["label"].tolist()

        # Initialize a sampler that will be used to efficiently sample batches of data
        self.sampler = None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        if self._targets is not None:
            return self._data[item], self._targets[item]
        else:
            return self._data[item]

    def get_dataloader(self, batch_size, max_len, pad_id):
        # Initialize the sampler to be used to efficiently sample batches of data
        self.sampler = SmartBatchingSampler(
            data_source=self._data, batch_size=batch_size
        )

        # Initialize a collate function that will be used to pad sequences to a fixed length and create batches
        collate_fn = SmartBatchingCollate(
            targets=self._targets, max_length=max_len, pad_token_id=pad_id
        )

        # Initialize a dataloader to iterate over batches of data
        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            sampler=self.sampler,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        return dataloader
