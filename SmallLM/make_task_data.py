"""Make synthetic tasks with checkable answers, for SFT, GRPO and evaluation.

Example:
    python make_task_data.py --data data/shakespeare_char

Words are taken from the dataset's own text, so every character is one the
tokenizer knows. Writes data/tasks/train.jsonl and data/tasks/test.jsonl;
test words never appear in train, so test accuracy measures real skill,
not memorization. See slm/tasks.py for the tasks.
"""

import argparse
import os
import random

from slm.data import load_tokenizer, load_tokens, save_jsonl
from slm.rl.rewards import words_in
from slm.tasks import TASKS, make_examples


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="dataset directory made by prepare.py")
    parser.add_argument("--out", default="data/tasks")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, help="default: all")
    parser.add_argument("--train", type=int, default=20000, help="number of training examples")
    parser.add_argument("--test", type=int, default=200, help="number of test examples")
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.data)
    text = tokenizer.decode(load_tokens(args.data, "train", tokenizer.vocab_size).tolist())
    words = sorted({w for w in words_in(text) if w.isalpha() and args.min_len <= len(w) <= args.max_len})

    # Keep test words out of the training set.
    random.Random(args.seed).shuffle(words)
    n_test_words = max(1, len(words) // 10)
    test_words, train_words = words[:n_test_words], words[n_test_words:]

    os.makedirs(args.out, exist_ok=True)
    save_jsonl(os.path.join(args.out, "train.jsonl"), make_examples(train_words, args.train, args.tasks, args.seed))
    save_jsonl(os.path.join(args.out, "test.jsonl"), make_examples(test_words, args.test, args.tasks, args.seed + 1))
    print(f"{len(train_words)} training words, {len(test_words)} test words -> {args.out}/")


if __name__ == "__main__":
    main()
