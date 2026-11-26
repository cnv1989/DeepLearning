"""Step 1: turn raw text into token ids ready for pretraining.

Examples:
    python prepare.py --dataset shakespeare --tokenizer char
    python prepare.py --dataset tinystories --tokenizer bpe --vocab-size 4096
    python prepare.py --input my_corpus.txt --tokenizer gpt2
    python prepare.py --input folder_of_txt_files/ --tokenizer bpe
    python prepare.py --input documents.jsonl --tokenizer bpe     (one {"text": ...} per line)

Pipeline:
    read documents ─► clean ─► remove duplicates ─► train/choose tokenizer
                   ─► encode, with EOS between documents ─► train.bin / val.bin

Tokenizers:
    char   one token per character
    bpe    byte-level BPE trained on this data (see slm/bpe.py)
    gpt2   OpenAI's pretrained GPT-2 BPE (50,257 tokens)

Writes train.bin, val.bin and tokenizer.json to data/<name>_<tokenizer>/.
"""

import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
import requests

from slm.bpe import BPETokenizer
from slm.data import save_tokenizer, token_dtype
from slm.tokenizer import EOS, CharTokenizer, TiktokenTokenizer

DATASETS = {
    "shakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "separator": None,  # one long document
    },
    "tinystories": {
        # The ~19 MB validation split of TinyStories: plenty for a first run.
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
        "separator": EOS,  # stories are separated by <|endoftext|>
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="shakespeare")
    parser.add_argument("--input", help="your own .txt / .jsonl file, or a folder of .txt files")
    parser.add_argument("--separator", help="string that separates documents inside a .txt file")
    parser.add_argument("--tokenizer", choices=["char", "bpe", "gpt2"], default="char")
    parser.add_argument("--vocab-size", type=int, default=4096, help="for --tokenizer bpe")
    parser.add_argument("--tokenizer-sample-chars", type=int, default=5_000_000,
                        help="train the BPE tokenizer on at most this many characters")
    parser.add_argument("--min-chars", type=int, default=1, help="drop documents shorter than this")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--out", help="output directory (default: data/<name>_<tokenizer>)")
    return parser.parse_args()


# ----------------------------------------------------------------------------- #
# Reading
# ----------------------------------------------------------------------------- #


def download(name):
    path = os.path.join("data", f"{name}.txt")
    if not os.path.exists(path):
        print(f"downloading {name} ...")
        response = requests.get(DATASETS[name]["url"], timeout=300)
        response.raise_for_status()
        os.makedirs("data", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(response.text)
    return path


def read_text_file(path, separator):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return text.split(separator) if separator else [text]


def read_documents(args):
    """Return a list of documents (strings) from whichever input was given."""
    if not args.input:
        path = download(args.dataset)
        return read_text_file(path, args.separator or DATASETS[args.dataset]["separator"])
    if os.path.isdir(args.input):
        documents = []
        for path in sorted(glob.glob(os.path.join(args.input, "**", "*.txt"), recursive=True)):
            documents.extend(read_text_file(path, args.separator))
        return documents
    if args.input.endswith(".jsonl"):
        with open(args.input, encoding="utf-8") as f:
            return [json.loads(line)["text"] for line in f if line.strip()]
    return read_text_file(args.input, args.separator)


# ----------------------------------------------------------------------------- #
# Cleaning
# ----------------------------------------------------------------------------- #


def clean(document):
    """Normalize line endings and trim surrounding whitespace."""
    return document.replace("\r\n", "\n").replace("\r", "\n").strip()


def remove_duplicates(documents):
    """Drop exact duplicate documents (common in scraped data, and they hurt training)."""
    seen, unique = set(), []
    for document in documents:
        digest = hashlib.sha1(document.encode("utf-8")).digest()
        if digest not in seen:
            seen.add(digest)
            unique.append(document)
    return unique


# ----------------------------------------------------------------------------- #


def build_tokenizer(args, documents):
    if args.tokenizer == "char":
        return CharTokenizer.from_text("".join(documents))
    if args.tokenizer == "gpt2":
        return TiktokenTokenizer("gpt2")

    sample, size = [], 0
    for document in documents:
        if size >= args.tokenizer_sample_chars:
            break
        sample.append(document)
        size += len(document)
    print(f"training a {args.vocab_size}-token BPE tokenizer on {size:,} characters ...")
    start = time.time()
    tokenizer = BPETokenizer.train(EOS.join(sample), args.vocab_size, verbose=True)
    print(f"  done in {time.time() - start:.0f}s")
    return tokenizer


def main():
    args = parse_args()
    name = os.path.splitext(os.path.basename(os.path.normpath(args.input)))[0] if args.input else args.dataset
    out_dir = args.out or os.path.join("data", f"{name}_{args.tokenizer}")
    os.makedirs(out_dir, exist_ok=True)

    raw = read_documents(args)
    documents = remove_duplicates([d for d in map(clean, raw) if len(d) >= args.min_chars])
    total_chars = sum(len(d) for d in documents)
    print(f"{len(raw):,} documents read, {len(documents):,} kept after cleaning "
          f"and deduplication ({total_chars:,} characters)")

    tokenizer = build_tokenizer(args, documents)

    # Encode every document and put EOS after each, so the model learns where documents end.
    token_ids = []
    for document in documents:
        token_ids.extend(tokenizer.encode(document))
        token_ids.append(tokenizer.eos_id)
    token_ids = np.array(token_ids, dtype=token_dtype(tokenizer.vocab_size))

    split_at = int(len(token_ids) * (1 - args.val_fraction))
    token_ids[:split_at].tofile(os.path.join(out_dir, "train.bin"))
    token_ids[split_at:].tofile(os.path.join(out_dir, "val.bin"))
    save_tokenizer(tokenizer, out_dir)

    print(f"vocab {tokenizer.vocab_size:,} | {total_chars / len(token_ids):.2f} characters per token | "
          f"train {split_at:,} tokens | val {len(token_ids) - split_at:,} tokens -> {out_dir}")


if __name__ == "__main__":
    main()
