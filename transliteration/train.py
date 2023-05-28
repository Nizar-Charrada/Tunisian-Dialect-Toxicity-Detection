import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.model_selection import train_test_split
from utils import set_seed
from dataloader.dataloader import TransData, custom_collate
from dataloader.preprocess import get_data
from models.model import TransformerConfig, TransformerModel
from models.optimizer import NoamOpt
import yaml
import os
import pickle
from utils import ParamsNamespace

# -----------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
config_path = "transliteration\config\config.yaml"

# -----------------------------------------------------------------------------


class Lang:

    """Class to create vocabulary for source and target language
    Args:
        config (ParamsNamespace): configuration set
        data (list): list of words
        language (str): source or target language"""

    def __init__(self, config, data, language):
        self.char2index = {}
        self.index2char = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.nchars = 3
        self.language = language
        self.is_target = language == config.target
        self.data = data

    def create_vocab(self):
        for word in self.data:
            self.addChar(word)

    def addChar(self, word):
        for char in word.lower():
            if char not in self.char2index:
                self.char2index[char] = self.nchars
                self.index2char[self.nchars] = char
                self.nchars += 1

    def tokenize(self, sentence, is_target=False):
        if self.is_target:
            return (
                [1] + [self.char2index[word] for word in sentence.lower()] + [2]
            )  # 1 is SOS and 2 is EOS
        else:
            return [self.char2index[word] for word in sentence.lower()]


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch or evaluate.
    Args:
        data_iter (DataLoader): data iterator for the dataset to operate on
        model (TransformerModel): model to train or evaluate
        loss_compute (nn.Module): loss function to compute
        optimizer (NoamOpt): optimizer to use during training
        mode (str): train or eval mode
        accum_iter (int): number of gradient accumulation steps to perform
        train_state (TrainState): train state to track progress"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    
    # start training loop
    for i, (src, tgt) in enumerate(data_iter):
        if mode == "train":
            model.train()
        else:
            model.eval()

        src = src.to(device)
        tgt = tgt.to(device)

        out = model.forward(src, tgt[:-1])
        out = out.reshape(-1, out.shape[2])
        #compute loss and backpropagate
        loss = loss_compute(out, tgt[1:].reshape(-1))

        # loss_node = loss_node / accum_iter
        if mode == "train":
            loss.backward()
            train_state.step += 1
            train_state.samples += src.shape[0]
            train_state.tokens += (
                tgt != config.PAD_token
            ).sum() - 2  # total token - sos and eos token
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.optimizer.zero_grad()
                n_accum += 1
                train_state.accum_step += 1

        total_loss += loss.item()
        total_tokens += (tgt != config.PAD_token).sum() - 2
        tokens += (tgt != config.PAD_token).sum() - 2
        if i % config.trainer.row_log_interval == 1 and (mode == "train"):
            lr = optimizer.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.4f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss

    return total_loss / total_tokens, train_state


def training(config):
    """Main training loop"""

    if config.load_model:
        checkpoint = torch.load(
            os.path.join(config.output_dir, f"{config.source}_to_{config.target}.bin")
        )
        model.load_state_dict(torch.load(f"{config.output_dir}/{config.source}_to_{config.target}.bin"))

    min_loss = 99999
    train_loss_list = []
    test_loss_list = []
    trainstate = TrainState()
    for epoch in range(config.trainer.max_epochs):
        train_loss, trainstate = run_epoch(
            train_loader, model, criterion, optimizer, train_state=trainstate
        )
        test_loss, _ = run_epoch(test_loader, model, criterion, optimizer, mode="test")
        print(
            ("Epoch : %6d | Train Loss: %6.4f " + "| Test Loss: %6.4f ")
            % (epoch, train_loss, test_loss)
        )
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        if test_loss < min_loss:
            min_loss = test_loss
            # Create the output directory if it doesn't already exist
            os.makedirs(config.output_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                f"{config.output_dir}/{config.source}_to_{config.target}.bin",
            )
            print(f"Saving model checkpoint to {config.output_dir}.")
            # torch.save({
            #    'epoch': epoch,
            #    'model_state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.optimizer.state_dict()
            #    }, PATH)

    return trainstate, train_loss_list, test_loss_list


""" Training Loop """

if __name__ == "__main__":
    assert os.path.isfile(config_path), "No configuration file found at {}".format(
        config_path
    )
    with open(config_path) as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)

    config = ParamsNamespace(yaml_config)

    assert (
        config.source == "arabizi" or config.source == "arabic"
    ), "Source language not found, please choose arabizi or arabic"
    assert (
        config.target == "arabizi" or config.target == "arabic"
    ), "Target language not found, please choose arabizi or arabic"

    set_seed(config.seed)
    data = get_data()
    # initialize vocab for source and target
    target_vocab = Lang(config, data[config.target], config.target)
    source_vocab = Lang(config, data[config.source], config.source)
    # create vocab for source and target
    target_vocab.create_vocab()
    source_vocab.create_vocab()
    # split data into train and test
    trainset, testset = train_test_split(
        data, test_size=config.trainer.test_size, random_state=config.seed
    )
    traindata = TransData(
        trainset, config.source, source_vocab, config.target, target_vocab
    )
    testdata = TransData(
        testset, config.source, source_vocab, config.target, target_vocab
    )
    # create dataloader for train and test
    train_loader = DataLoader(
        traindata,
        batch_size=config.dataset.batch_size,
        shuffle=config.dataset.shuffle,
        collate_fn=custom_collate,
    )
    test_loader = DataLoader(
        testdata,
        batch_size=config.dataset.batch_size,
        shuffle=config.dataset.shuffle,
        collate_fn=custom_collate,
    )
    # model init
    model_args = TransformerConfig(**yaml_config["transformer"])

    model_args.src_vocab_size = len(source_vocab.index2char)
    model_args.tgt_vocab_size = len(target_vocab.index2char)

    model = TransformerModel(
        model_args,
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_token)
    optimizer = NoamOpt(
        config.transformer.hidden_dim,
        config.optim.factor,
        config.optim.warmup,
        optim.Adam(model.parameters(), lr=config.optim.lr),
    )

    # save vocab to pickle file for inference
    source_vocab_file_name = f"{config.source}_vocab.pkl"
    file_path = os.path.join(config.output_dir, source_vocab_file_name)
    with open(file_path, "wb") as file:
        pickle.dump(source_vocab, file)
        print(f'Vocab successfully saved to "{file_path}"')

    target_vocab_file_name = f"{config.target}_vocab.pkl"
    file_path = os.path.join(config.output_dir, target_vocab_file_name)
    with open(file_path, "wb") as file:
        pickle.dump(target_vocab, file)
        print(f'Vocab successfully saved to "{file_path}"')
    # train model
    trainstate, train_loss_list, test_loss_list = training(config)
