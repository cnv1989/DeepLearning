# 7. Supervised fine-tuning (SFT)

**Code:** [`train_sft.py`](../train_sft.py), [`slm/sft.py`](../slm/sft.py), [`make_task_data.py`](../make_task_data.py), [`slm/tasks.py`](../slm/tasks.py)

## What it is

A pretrained model **continues text**. Ask it "What is the capital of France?"
and it might continue with "What is the capital of Spain?", because lists of
questions are common on the web. It isn't trying to *answer*.

SFT fixes this by training on examples of the behaviour we want: a prompt
followed by a good reply. It's the same next-token training as pretraining,
with two changes: the data is conversations, and only the replies are graded.

```bash
python make_task_data.py --data data/shakespeare_char
python train_sft.py --ckpt out/ckpt.pt --train-file data/tasks/train.jsonl --max-iters 1500
python evaluate.py --ckpt out/sft/ckpt.pt --tasks data/tasks/test.jsonl
```

## Data format

JSON lines, one example per line, in either form:

```json
{"prompt": "Uppercase: rocky", "response": "ROCKY"}
{"messages": [{"role": "system", "content": "Be brief."},
              {"role": "user", "content": "Hi"},
              {"role": "assistant", "content": "Hello!"},
              {"role": "user", "content": "Reverse: rose"},
              {"role": "assistant", "content": "esor"}]}
```

`"answer"` also works in place of `"response"`, so task files can be used
directly.

## The chat template (`render_prompt`, `tokenize_conversation`)

The model only understands one flat sequence of tokens, so conversations are
written out in a fixed pattern:

```
User: Uppercase: rocky
Assistant: ROCKY<|endoftext|>
```

The `User:` / `Assistant:` labels tell the model whose turn it is. The EOS
token after the reply teaches it to **stop**. At generation time, the prompt is
rendered the same way and ends with `Assistant: `, so the model's natural next
move is to write a reply.

The template must be identical at training and generation time. A test
(`test_prompt_matches_training_format`) checks that the generation prompt is
exactly the beginning of a training example. Real models use special tokens
like `<|im_start|>` instead of plain-text labels, but the idea is the same.

## Loss masking: train on the answer only

```
tokens:    User: Uppercase: rocky\nAssistant: ROCKY<EOS>\n
trained:   ───────────── no ──────────────────  ─yes──  no
```

Every position the model shouldn't be graded on gets target **−100**, which
`cross_entropy` skips (`ignore_index=-100`). The model still *reads* the user's
text (it has to, to answer), but it isn't trained to *produce* it. Without
masking, a lot of capacity goes into predicting user questions, which isn't
its job.

`SFTDataset.get_batch` pads examples of different lengths into one batch. Padding
positions also get −100, so they never affect the loss.

## Synthetic tasks (`make_task_data.py`, `slm/tasks.py`)

To watch SFT work on a tiny model, the repo generates tasks with exactly one
right answer:

| Task | Prompt | Answer |
|---|---|---|
| uppercase | `Uppercase: rocky` | `ROCKY` |
| reverse | `Reverse: convoy` | `yovnoc` |
| repeat | `Repeat: cat` | `cat cat` |
| first letter | `First letter: slide` | `s` |

Words come from the dataset's own text, so the character tokenizer knows every
letter. **Test words never appear in training**, so test accuracy measures
whether the model learned the *skill* (how to reverse), not memorized
answers.

## Results from a test run

Starting from the 6-minute pretrained `tiny` model:

| Fine-tuning steps | Test accuracy (unseen words) |
|---|---|
| 0 (just pretrained) | 0%: it writes Shakespeare instead of answering |
| 250 | 5.5% |
| 500 | 23.5%: "first letter" is learned first (98%) |
| 1,500 (2 minutes) | **100%** on all four tasks |

The easy task (first letter) is learned first, and "reverse", which needs the
model to track positions, comes last.

## How SFT differs from pretraining

| | Pretraining | SFT |
|---|---|---|
| Data | huge, unlabeled text | small, curated examples |
| Loss on | every token | reply tokens only |
| Learning rate | higher (1e-3 here) | lower (3e-4); don't wreck what was learned |
| Weight decay | 0.1 | 0 |
| Steps | many | few |

Both scripts share one training loop (`slm/trainer.py`); only the batches
differ. SFT also supports `--resume`, multiple GPUs, mixed precision and
everything else from [Pretraining](05-pretraining.md).

## Common problems

* **The model never stops**: EOS isn't at the end of replies, or examples were
  cut off by `block_size` (those with no reply left are skipped).
* **It forgets general knowledge** ("catastrophic forgetting"): too many steps
  or too high a learning rate. Mix in some general data, or train less.
* **Great on training prompts, bad on new ones**: overfitting to a small
  dataset. Watch validation loss; more varied examples help.
* **Answers in the wrong format**: the generation prompt doesn't match the
  training template exactly.

## Try it

```bash
python make_task_data.py --data data/shakespeare_char --tasks reverse --max-len 10
```

Longer words make reversing much harder. How many steps does the model need
now?
