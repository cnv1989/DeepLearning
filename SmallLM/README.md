# SmallLM

A complete pipeline for training a language model, built from scratch in
PyTorch and written to be read. It covers every stage modern LLMs go through,
from a tokenizer to reinforcement learning, and the smallest version trains on
a laptop CPU in minutes.

```
  raw text
     │  prepare.py ─────────── clean, deduplicate, train a tokenizer, encode
     ▼
  token ids
     │  train.py ───────────── PRETRAINING: predict the next token
     ▼
  base model      out/ckpt.pt
     │  train_sft.py ───────── SUPERVISED FINE-TUNING: learn to answer prompts
     ▼
  chat model      out/sft/ckpt.pt
     │  train_grpo.py ──────── REINFORCEMENT LEARNING against a reward, or
     │  train_dpo.py ───────── PREFERENCE LEARNING from (chosen, rejected) pairs
     ▼
  tuned model     out/grpo.pt, out/dpo.pt
                  evaluate.py, sample.py work with every checkpoint
```

**New to this?** The [guides in `docs/`](docs/README.md) explain every component
(tokenization, the model, pretraining, fine-tuning, RL and more) step by step.

## Quick start (CPU, about 10 minutes)

```bash
cd SmallLM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python prepare.py --dataset shakespeare --tokenizer char         # 1. data
python train.py --config configs/tiny.json --data data/shakespeare_char   # 2. pretrain (~6 min)
python sample.py --prompt "ROMEO:"                                # generate text

python make_task_data.py --data data/shakespeare_char           # 3. tasks with known answers
python train_sft.py --ckpt out/ckpt.pt --train-file data/tasks/train.jsonl --max-iters 1500
python evaluate.py --ckpt out/sft/ckpt.pt --tasks data/tasks/test.jsonl
python sample.py --ckpt out/sft/ckpt.pt --chat --prompt "Reverse: flower" --temperature 0
```

### Results from a test run (4-core CPU)

| Stage | Result |
|---|---|
| Pretrain `tiny` (0.86M params, 1,500 steps, 6.5 min) | validation loss **1.59**, perplexity 4.9 (the earlier GPT-2-style model reached 1.91 at the same size) |
| SFT on synthetic tasks (1,500 steps, 2 min) | accuracy on unseen words **0% → 100%** (uppercase, reverse, repeat, first letter) |
| GRPO on open-ended text (100 steps) | reward 1.78 → 1.83 (out of 2.0) |
| GRPO on tasks, from a half-trained SFT model | stable, but no accuracy gain in 400 steps (23.5% → 22.5%); see [GRPO tips](#grpo-tips) |

Pretraining output after 6 minutes:

```
ROMEO:
He good to hear their all-would have shaped that
We did if an to our kiss in tell of that.
```

## The stages

### 1. Data and tokenizers (`prepare.py`)

```bash
python prepare.py --dataset shakespeare --tokenizer char
python prepare.py --dataset tinystories --tokenizer bpe --vocab-size 4096
python prepare.py --input my_corpus.txt --tokenizer gpt2
python prepare.py --input folder_of_txt_files/ --tokenizer bpe
python prepare.py --input documents.jsonl --tokenizer bpe      # one {"text": ...} per line
```

Documents are cleaned, exact duplicates removed, and an end-of-text token (EOS)
placed after each one so the model learns where documents end.

| `--tokenizer` | What it is |
|---|---|
| `char` | one token per character |
| `bpe` | byte-level BPE **trained on your data** (`slm/bpe.py`), the same algorithm GPT-2 and Llama use. 2 seconds for a 2,048-token vocabulary on Shakespeare, giving 2.9 characters per token |
| `gpt2` | OpenAI's pretrained 50,257-token BPE via `tiktoken` |

### 2. Pretraining (`train.py`)

```bash
python train.py --config configs/tiny.json --data data/shakespeare_char           # CPU
python train.py --config configs/small.json --data data/tinystories_bpe           # one GPU
torchrun --nproc_per_node 4 train.py --config configs/medium.json --data ...      # 4 GPUs
```

| Feature | How |
|---|---|
| Configs | `configs/*.json`; any setting can also be given as a flag (`--max-iters 3000`) |
| Multi-GPU | launch with `torchrun`; uses DistributedDataParallel (`slm/distributed.py`) |
| Mixed precision | `--dtype auto` (bfloat16 on new GPUs, float16 with loss scaling on older ones) |
| Big batches on small GPUs | `--grad-accum 8` accumulates gradients over 8 micro-batches |
| Less memory | `--activation-checkpointing` recomputes activations instead of storing them |
| Speed | `--compile` (`torch.compile`) on GPUs |
| Checkpoints | `out/ckpt.pt` (best), `out/last.pt` (latest); continue with `--resume` |
| Metrics | `out/metrics.jsonl` every step; add `--wandb` for Weights & Biases |

### 3. Supervised fine-tuning (`train_sft.py`)

Teaches the model to answer. Data is JSON lines of `{"prompt", "response"}`,
or `{"messages": [{"role", "content"}, ...]}` for multi-turn chats. It's
rendered with a simple chat template:

```
User: Reverse: flower
Assistant: rewolf<|endoftext|>
```

Only the assistant's reply is trained on (loss masking). `make_task_data.py`
generates tasks with checkable answers, with test words kept out of training.

### 4. Reinforcement learning

**GRPO** (`train_grpo.py`): for each prompt, sample a group of answers, score them,
and push the model toward the above-average ones. A KL penalty keeps it close
to where it started.

```bash
# Verifiable rewards: task prompts with known answers
python train_grpo.py --ckpt out/sft/ckpt.pt --prompts data/tasks/train.jsonl \
    --reward exact_match --reward similarity --lr 3e-5

# Open-ended text with rule-based rewards
python train_grpo.py --ckpt out/ckpt.pt --data data/shakespeare_char \
    --reward known_words --reward distinct_words --lr 3e-5
```

| Reward | Scores |
|---|---|
| `exact_match` | 1 if the answer is exactly right |
| `similarity` | 0–1 partial credit for being close to the answer |
| `known_words` | fraction of words that appear in the training text |
| `distinct_words` | fraction of words that aren't repeats |
| `keyword=love` | 1 if the word appears |
| `model=out/reward_model.pt` | a learned reward model |

**Preferences, reward models and DPO**: real RLHF starts from people choosing
the better of two answers. `make_preferences.py` uses a rule-based reward as
the judge; you can also write `{"prompt", "chosen", "rejected"}` lines yourself.

```bash
python make_preferences.py --data data/shakespeare_char --reward known_words --reward distinct_words
python train_reward_model.py            # then: train_grpo.py --reward model=out/reward_model.pt
python train_dpo.py --lr 3e-5           # or skip the reward model entirely with DPO
```

#### GRPO tips

* **The model must sometimes succeed.** GRPO learns from differences within a
  group. If all 8 attempts fail, there's nothing to learn. Start from a decent
  SFT model, and add `similarity` for partial credit.
* **Keep the learning rate low.** On the tiny model, 1e-4 moved it 1.4 nats per
  token from the reference in a single step. 3e-5 is stable. Watch `kl`.
* **Watch for reward hacking.** With only `known_words`, the model learns to
  write "the the the…". `distinct_words` closes that loophole. Read samples.
* In the test run, longer SFT solved the tasks outright, so RL had nothing
  left to fix. RL pays off when the reward captures something SFT data can't.

### 5. Evaluation and generation

```bash
python evaluate.py --ckpt out/ckpt.pt --data data/shakespeare_char      # perplexity
python evaluate.py --ckpt out/sft/ckpt.pt --tasks data/tasks/test.jsonl # accuracy per task
python sample.py --ckpt out/sft/ckpt.pt --chat                          # interactive chat
```

## The model (`slm/model.py`)

A Llama-style decoder-only transformer:

| Component | Why |
|---|---|
| RMSNorm | simpler and cheaper than LayerNorm |
| Rotary position embeddings (RoPE) | encode relative position directly in attention |
| SwiGLU feed-forward | trains better than a GELU MLP at the same size |
| Grouped-query attention (`n_kv_head`) | query heads share key/value heads, shrinking the KV cache |
| KV cache | generation reuses earlier keys/values, so each new token costs one position |
| Tied embeddings | input and output share one matrix, saving parameters |

| Config | Layers | Heads (query/kv) | Width | Context | Params (char / 4k BPE / GPT-2 vocab) |
|---|---|---|---|---|---|
| `tiny` | 4 | 4 / 4 | 128 | 128 | 0.86M / 1.4M / 7.3M |
| `small` | 6 | 6 / 2 | 384 | 256 | 9.5M / 11M / 29M |
| `medium` | 12 | 12 / 4 | 768 | 1024 | 76M / 79M / 114M |

`small` is a good size for a single GPU, and `medium` for several (it uses `--compile` and `--grad-accum 8`).

## Code map

| File | What it does |
|---|---|
| **Scripts** | |
| `prepare.py` | Read, clean and deduplicate text; train or pick a tokenizer; write `train.bin` / `val.bin` |
| `train.py` | Pretraining |
| `train_sft.py` | Supervised fine-tuning |
| `make_task_data.py` | Synthetic tasks with checkable answers |
| `train_grpo.py` | GRPO reinforcement learning |
| `make_preferences.py` | Preference pairs judged by a reward |
| `train_reward_model.py` | Learn a reward model from preferences |
| `train_dpo.py` | Direct Preference Optimization |
| `evaluate.py` | Perplexity and task accuracy |
| `sample.py` | Text generation and chat |
| **Library: `slm/`** | |
| `model.py` | The transformer, KV cache and generation |
| `tokenizer.py`, `bpe.py` | Character, trained-BPE and GPT-2 tokenizers; the BPE training algorithm |
| `data.py` | Load token files, sample batches, read/write JSON lines |
| `trainer.py` | The training loop shared by pretraining and SFT |
| `config.py` | Settings from defaults, JSON configs and flags |
| `distributed.py` | Multi-GPU setup |
| `metrics.py` | Metrics logging |
| `sft.py` | Chat template, loss masking, SFT batches, reply generation |
| `tasks.py` | Synthetic task definitions |
| `checkpoint.py`, `utils.py` | Save/load; device, precision and LR schedule |
| **RL: `slm/rl/`** | |
| `logprobs.py` | Token log-probabilities, completion masks, padding |
| `rewards.py` | Verifiable, rule-based and learned rewards |
| `grpo.py` | Group-relative advantages, clipped objective, KL penalty |
| `reward_model.py`, `dpo.py`, `preferences.py` | Reward model, DPO loss, preference data |
| **Tests: `tests/`** | `pytest` (21 tests: tokenizers, KV cache, causality, loss masking, RL losses) |

For explanations of each part, see the [guides](docs/README.md). Suggested code reading order: `slm/model.py` → `train.py` + `slm/trainer.py` →
`slm/sft.py` → `slm/rl/grpo.py` + `train_grpo.py`.

## Tests

```bash
python -m pytest
```
