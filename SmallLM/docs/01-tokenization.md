# 1. Tokenization

**Code:** [`slm/tokenizer.py`](../slm/tokenizer.py), [`slm/bpe.py`](../slm/bpe.py) · **Tests:** [`tests/test_tokenizers.py`](../tests/test_tokenizers.py)

## What it is

A neural network works with numbers, not letters. A **tokenizer** converts text
into a list of integers (**token ids**) and back again:

```
"the cat"  ──encode──►  [1169, 3797]  ──decode──►  "the cat"        (GPT-2 tokenizer)
```

The set of all possible tokens is the **vocabulary**, and its size is
`vocab_size`. The model's first layer looks up a vector for each id, and its
last layer outputs one score per vocabulary entry.

## Why it matters

The tokenizer decides what the model "sees" as one unit, and that choice has
real consequences:

| | Character tokens | Word-piece (BPE) tokens |
|---|---|---|
| Vocabulary | tiny (~66 for Shakespeare) | thousands (2k–100k) |
| Tokens per sentence | many (one per character) | few (~1 per 3–4 characters) |
| Context window covers | less text | more text |
| Model must learn | spelling *and* meaning | mostly meaning |
| Unknown characters | dropped | never happen (byte-level) |

Fewer tokens per sentence means the model sees more text per step and can
look further back, which is why real LLMs use BPE. Character tokens are easier
to understand and work well for tiny experiments.

## The three tokenizers

| `prepare.py --tokenizer` | Class | Vocabulary |
|---|---|---|
| `char` | `CharTokenizer` | every distinct character in the data, plus EOS |
| `bpe` | `BPETokenizer` | 256 bytes + learned merges + EOS, trained on *your* data |
| `gpt2` | `TiktokenTokenizer` | OpenAI's fixed 50,257-token GPT-2 vocabulary |

All three share one interface: `encode(text) -> ids`, `decode(ids) -> text`,
`vocab_size`, `eos_id` and `to_dict()`. That's why everything else in the code
works with any of them.

## The end-of-text token (EOS)

Every tokenizer reserves one special id for `<|endoftext|>`. It has two jobs:

1. **In pretraining**, it separates documents, so the model learns that a story
   ended and the next text is unrelated.
2. **In fine-tuning**, it ends every answer, so the model learns **when to
   stop**. Generation halts when it produces EOS.

The literal string `"<|endoftext|>"` in text becomes that one id (see
`split_on_eos`), rather than 13 characters.

## How byte-level BPE works

BPE (byte-pair encoding) builds a vocabulary by repeatedly merging the most
common neighbouring pair of tokens.

**Start:** text is converted to UTF-8 bytes. The initial vocabulary is the 256
possible byte values, so *any* string, in any language or with emoji, can be
represented. There are never unknown characters.

**Train** (`train_bpe`):

```
corpus:  "the the then"

step 0   t h e · t h e · t h e n          most common pair: (t, h)
step 1   th e · th e · th e n             merge → new token 256 = "th"
step 2   the · the · the n                merge (th, e) → 257 = "the"
step 3   ...                              continue until vocab_size is reached
```

The output is an **ordered list of merges**. That list *is* the tokenizer.

**Encode** (`BPETokenizer._encode_chunk`): start from bytes and replay the
merges, always applying the one learned **earliest** first, until no learned
merge applies.

**Pre-tokenization.** Before counting pairs, text is split into chunks
(words with their leading space, numbers, punctuation, whitespace) by
`CHUNK_PATTERN`. Merges never cross a chunk boundary, so you get tokens like
`" light"`, never `"t. Th"`. The pattern is built so its pieces always join
back into the original text exactly (a test checks this).

**Speed tricks** in `train_bpe`:

* Each distinct chunk is counted once with its frequency, so the work depends
  on the number of *distinct* words, not the length of the text.
* Pair counts are updated incrementally, only for the words that contain the
  pair just merged, instead of recounting everything.

On Shakespeare (1.1M characters), learning 2,048 tokens takes about 2 seconds:

```
"ROMEO: But soft, what light through yonder window breaks?"   (57 characters)
→ ['ROMEO', ':', ' But', ' so', 'ft', ',', ' what', ' light', ' through',
   ' y', 'onder', ' wind', 'ow', ' break', ...]                (16 tokens)
```

Common words become single tokens; rarer ones split into pieces.

## Choosing a vocabulary size

* **Too small:** text becomes long sequences of tiny pieces, so attention has to
  do more work to reassemble words.
* **Too large:** the embedding table (`vocab_size × n_embd`) dominates a small
  model's parameters, and rare tokens are seen too few times to learn well.

For the `tiny` model, the embedding is 66 × 128 = 8k parameters with characters
but 50,257 × 128 = 6.4M with the GPT-2 vocabulary: 7× the rest of the model.
**For small models, train your own BPE with 2k–8k tokens.**

## Saving

The tokenizer is saved as `tokenizer.json` next to the data, and is also stored
**inside every checkpoint** (`slm/checkpoint.py`). A checkpoint is useless
without the exact tokenizer it was trained with, so keeping them together
prevents mix-ups.

## Common problems

* **Characters missing from output**: `CharTokenizer` silently drops characters
  it never saw in training. Use `bpe`, or include those characters in the data.
* **Different tokenizer at train and test time**: always load the tokenizer from
  the checkpoint, as every script here does.
* **Tokenizing pieces separately vs together**: BPE can split `"ab"` differently
  from `"a"` + `"b"`. The fine-tuning code tokenizes the chat template's pieces
  the same way at training and generation time for this reason.

## Try it

```bash
python prepare.py --dataset shakespeare --tokenizer bpe --vocab-size 512
python prepare.py --dataset shakespeare --tokenizer bpe --vocab-size 8192
```

Compare the "characters per token" each prints. Then train the same model on
each and compare samples.
