"""Preference data: which of two completions is better.

Stored as JSON lines, one pair per line:

    {"prompt": "ROMEO:\n", "chosen": "What light ...", "rejected": "Thhe tthe ..."}

You can write this file by hand, collect it from people, or generate it
automatically with make_preferences.py.
"""

import random

from slm.data import load_jsonl, save_jsonl


def save_preferences(path, pairs):
    save_jsonl(path, pairs)


def load_preferences(path):
    return load_jsonl(path)


def split_preferences(pairs, val_fraction=0.1, seed=0):
    """Shuffle, then hold out a fraction of the pairs for validation."""
    pairs = pairs[:]
    random.Random(seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_fraction))
    return pairs[n_val:], pairs[:n_val]


def batches(items, batch_size, shuffle=True):
    order = list(range(len(items)))
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [items[i] for i in order[start : start + batch_size]]
