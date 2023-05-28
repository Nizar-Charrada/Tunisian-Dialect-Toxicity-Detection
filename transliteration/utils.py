import random
import numpy as np
import torch
import os
import argparse
import re
import pyarabic.araby as araby

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


class ParamsNamespace(argparse.Namespace):
    """A namespace object that allows for dot notation access to the parameters."""

    def __init__(self, params_dict):
        for key, value in params_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ParamsNamespace(value))
            else:
                setattr(self, key, value)


def detect_arabic_type(text):
    """Detect if the text is Arabic, Arabizi, or both."""
    words = re.findall(r"[\w']+|[?!.,]", text)
    is_arabic = False
    is_arabizi = False
    for word in words:
        if araby.is_arabicrange(word):
            is_arabic = True
        else:
            is_arabizi = True

    if is_arabic and not is_arabizi:
        return "Arabic"
    elif not is_arabic and is_arabizi:
        return "Arabizi"
    else:
        return "Both Arabic and Arabizi"
