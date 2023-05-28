import yaml
import os
import torch
import shutil
import numpy as np
import random
import argparse

class ParamsNamespace(argparse.Namespace):
    """A namespace object that allows for dot notation access to the parameters."""

    def __init__(self, params_dict):
        for key, value in params_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ParamsNamespace(value))
            else:
                setattr(self, key, value)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, "last.pth.tar")
    if not os.path.exists(checkpoint):
        print(
            "Checkpoint Directory does not exist! Making directory {}".format(
                checkpoint
            )
        )
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "best.pth.tar"))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optim_dict"])

    return checkpoint



def fix_all_seeds(seed):
    """Fixes all seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val

def load_config(config_path):
    assert os.path.isfile(config_path), "No configuration file found at {}".format(
        config_path
    )
    with open(config_path) as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_config

def remove_stop_words(text, stop_words):
    """Removes stop words from text and returns the text"""
    text_list = text.split(" ")
    new_text = []
    for word in text_list:
        if word not in stop_words:
            new_text.append(word)

    return " ".join(new_text)