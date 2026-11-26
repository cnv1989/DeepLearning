"""Preference data: which of two completions is better.

Stored as JSON lines, one pair per line:

    {"prompt": "ROMEO:\n", "chosen": "What light ...", "rejected": "Thhe tthe ..."}

You can write this file by hand, collect it from people, or generate it
automatically with make_preferences.py.
"""

import json
import random


def save_preferences(path, pairs):
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")


def load_preferences(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
