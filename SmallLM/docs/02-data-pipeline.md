# 2. The data pipeline

**Code:** [`prepare.py`](../prepare.py), [`slm/data.py`](../slm/data.py)

## What it is

Pretraining needs one thing: a very long list of token ids. The data pipeline
turns raw text, wherever it comes from, into that list, saved to disk in a form
that's fast to read during training.

```
 read ──► clean ──► deduplicate ──► tokenizer ──► encode + EOS ──► split ──► train.bin
                                                                            val.bin
                                                                            tokenizer.json
```

## Why it matters

"Garbage in, garbage out" applies more to language models than almost anything
else. Data quality usually matters more than model tweaks. Real LLM teams spend
most of their effort here: filtering spam, removing duplicates, balancing
sources. This pipeline shows the basic steps that every serious pipeline
builds on.

## Step by step

### 1. Read documents (`read_documents`)

A **document** is one self-contained piece of text: a story, an article, a
book. Inputs can be:

| Input | Becomes |
|---|---|
| `--dataset shakespeare` | downloaded, then one long document |
| `--dataset tinystories` | downloaded, split on `<\|endoftext\|>` into separate stories |
| `--input file.txt` | one document, or several if you pass `--separator` |
| `--input folder/` | every `.txt` file inside it, recursively |
| `--input file.jsonl` | one document per line, from its `"text"` field |

### 2. Clean (`clean`)

Normalizes Windows/Mac line endings to `\n` and trims whitespace. Documents
shorter than `--min-chars` are dropped. Real pipelines go much further:
language detection, removing boilerplate such as navigation menus, filtering
toxic content, and scoring quality.

### 3. Remove duplicates (`remove_duplicates`)

Each document is hashed (SHA-1) and repeats are dropped. Web data is full of
copies, such as licence texts and repeated posts. Duplicates make the model
memorize instead of generalize, and waste compute. This is *exact*
deduplication; production pipelines also remove *near*-duplicates (for example
with MinHash).

### 4. Build the tokenizer (`build_tokenizer`)

Either built from the characters present, trained as BPE on up to
`--tokenizer-sample-chars` characters (5M by default; a sample is enough), or
the fixed GPT-2 vocabulary. See [Tokenization](01-tokenization.md).

### 5. Encode, with EOS between documents

```
doc 1 tokens, EOS, doc 2 tokens, EOS, doc 3 tokens, EOS, ...
```

Everything is concatenated into one long stream. The EOS tokens tell the model
where one document ends, so it doesn't learn false connections between
unrelated texts.

### 6. Split and save

The last `--val-fraction` (10%) becomes the **validation set**: text the model
never trains on, used to check whether it's learning general patterns or just
memorizing. Tokens are saved as raw `uint16` numbers (`uint32` if the
vocabulary exceeds 65,535), 2 bytes per token, no headers.

## Reading data during training (`slm/data.py`)

### Memory mapping

`load_tokens` opens `train.bin` with `np.memmap`. The file isn't loaded into
RAM; the operating system reads just the parts that are touched. A 100 GB
dataset works on a laptop with 16 GB of memory.

### Random windows

`get_batch` picks `batch_size` random starting points and cuts out
`block_size + 1` consecutive tokens at each:

```
tokens:   ... 41  7 19 88  3 52 ...
window:       41  7 19 88  3
inputs:       41  7 19 88          (what the model reads)
targets:       7 19 88  3          (what it should predict: shifted by one)
```

Every position is a training example: after `41` predict `7`, after `41 7`
predict `19`, and so on. One window of 128 tokens gives 128 predictions at
once.

Random sampling means there are no "epochs" in the usual sense. Over enough
steps the model sees the data roughly uniformly. With 1M training tokens,
`batch_size` 32 and `block_size` 128, each step covers 4,096 tokens, so 1,500
steps sees about 6M tokens: each token about 6 times.

## How much data do you need?

A common rule of thumb from the Chinchilla scaling study is about **20 tokens
per parameter** for compute-efficient training.

| Model | Parameters | ~Tokens wanted | Example data |
|---|---|---|---|
| `tiny` | 0.9M | 18M | Shakespeare (1M) seen many times |
| `small` | 10M | 200M | the full TinyStories training set (a few hundred million) |
| `medium` | 80M | 1.6B | a slice of a web dataset like FineWeb |

With less data than that, the model will start to **overfit**: training loss
keeps falling while validation loss stops improving or rises. Dropout and
weight decay help a little; more data helps a lot.

## Common problems

* **Validation loss much higher than training loss**: overfitting. Get more
  data, use a smaller model, or raise `dropout`.
* **Leakage**: if near-copies of validation text are also in training, the
  validation loss is falsely good. Deduplicate *before* splitting, as here.
* **Wrong separator**: if documents aren't split, EOS tokens never appear and
  the model never learns to stop.

## Try it

```bash
python prepare.py --dataset tinystories --tokenizer bpe --vocab-size 4096
```

Look at the output: documents read, documents kept after deduplication,
characters per token and token counts.
