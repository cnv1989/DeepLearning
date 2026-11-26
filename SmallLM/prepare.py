"""Step 1: download a text dataset, tokenize it, and save it for training.

Examples:
    python prepare.py --dataset shakespeare --tokenizer char
    python prepare.py --dataset tinystories --tokenizer bpe
    python prepare.py --input my_corpus.txt --tokenizer bpe

Writes train.bin, val.bin and tokenizer.json to data/<name>_<tokenizer>/.
"""

import argparse
import os

import numpy as np
import requests

from slm.data import save_tokenizer, token_dtype
from slm.tokenizer import BPETokenizer, CharTokenizer

DATASET_URLS = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    # The ~19 MB validation split of TinyStories: plenty for a first run.
    "tinystories": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASET_URLS, default="shakespeare")
    parser.add_argument("--input", help="path to your own .txt file (used instead of --dataset)")
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    parser.add_argument("--out", help="output directory (default: data/<name>_<tokenizer>)")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    return parser.parse_args()


def read_text(args):
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            return f.read()

    cached_path = os.path.join("data", f"{args.dataset}.txt")
    if not os.path.exists(cached_path):
        print(f"downloading {args.dataset} ...")
        response = requests.get(DATASET_URLS[args.dataset], timeout=120)
        response.raise_for_status()
        os.makedirs("data", exist_ok=True)
        with open(cached_path, "w", encoding="utf-8") as f:
            f.write(response.text)
    with open(cached_path, encoding="utf-8") as f:
        return f.read()


def main():
    args = parse_args()
    name = os.path.splitext(os.path.basename(args.input))[0] if args.input else args.dataset
    out_dir = args.out or os.path.join("data", f"{name}_{args.tokenizer}")
    os.makedirs(out_dir, exist_ok=True)

    text = read_text(args)
    print(f"corpus: {len(text):,} characters")

    tokenizer = CharTokenizer.from_text(text) if args.tokenizer == "char" else BPETokenizer()
    token_ids = np.array(tokenizer.encode(text), dtype=token_dtype(tokenizer.vocab_size))

    split_at = int(len(token_ids) * (1 - args.val_fraction))
    token_ids[:split_at].tofile(os.path.join(out_dir, "train.bin"))
    token_ids[split_at:].tofile(os.path.join(out_dir, "val.bin"))
    save_tokenizer(tokenizer, out_dir)

    print(f"vocab {tokenizer.vocab_size:,} | train {split_at:,} tokens | "
          f"val {len(token_ids) - split_at:,} tokens -> {out_dir}")


if __name__ == "__main__":
    main()
