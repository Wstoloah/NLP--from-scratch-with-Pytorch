import random
import string
import unicodedata
import numpy as np
import torch

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Constants
allowed_characters = string.ascii_letters + " .,;'" + "_"
n_letters = len(allowed_characters)

def unicodeToAscii(s):
    """Convert unicode string to ASCII"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in allowed_characters
    )

def letterToIndex(letter):
    """Find letter index from allowed_characters"""
    if letter not in allowed_characters:
        return allowed_characters.find("_")  # OOV character
    else:
        return allowed_characters.find(letter)

def lineToTensor(line):
    """Convert a line to a tensor of one-hot letter vectors"""
    tensor = torch.zeros(len(line), n_letters)  # Shape: (seq_len, n_letters)
    for li, letter in enumerate(line):
        tensor[li][letterToIndex(letter)] = 1
    return tensor
