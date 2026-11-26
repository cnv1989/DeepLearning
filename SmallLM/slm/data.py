"""Reading tokenized datasets made by prepare.py.

A dataset directory contains:
    train.bin       all training token ids, one after another
    val.bin         held-out token ids for measuring progress
    tokenizer.json  how to turn ids back into text
"""

import json
import os

import numpy as np
import torch

from slm.tokenizer import tokenizer_from_dict


def token_dtype(vocab_size):
    """Smallest numpy integer type that can hold every token id."""
    return np.uint16 if vocab_size < 2**16 else np.uint32


def save_tokenizer(tokenizer, data_dir):
    with open(os.path.join(data_dir, "tokenizer.json"), "w") as f:
        json.dump(tokenizer.to_dict(), f)


def load_tokenizer(data_dir):
    with open(os.path.join(data_dir, "tokenizer.json")) as f:
        return tokenizer_from_dict(json.load(f))


def load_tokens(data_dir, split, vocab_size):
    """Memory-map train.bin or val.bin, so large datasets aren't loaded into RAM."""
    path = os.path.join(data_dir, f"{split}.bin")
    return np.memmap(path, dtype=token_dtype(vocab_size), mode="r")


def random_windows(tokens, count, length, device):
    """Pick `count` random slices of `length` consecutive tokens.

    Returns a LongTensor of shape (count, length).
    """
    starts = torch.randint(len(tokens) - length, (count,)).tolist()
    windows = np.stack([tokens[s : s + length] for s in starts]).astype(np.int64)
    return torch.from_numpy(windows).to(device)


def get_batch(tokens, batch_size, block_size, device):
    """A batch for next-token prediction.

    inputs  = tokens[i     : i + block_size]
    targets = tokens[i + 1 : i + block_size + 1]   (each input shifted by one)
    """
    windows = random_windows(tokens, batch_size, block_size + 1, device)
    return windows[:, :-1], windows[:, 1:]
