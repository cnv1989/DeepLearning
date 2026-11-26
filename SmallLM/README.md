# SmallLM

A small GPT-style language model built from scratch in PyTorch, plus the
reinforcement-learning (RL) stages used to fine-tune modern language models.
The code is written to be read: short files, plain names, and comments that
explain *why* as well as *what*.

```
            pretraining                         reinforcement learning
  text ──► prepare.py ──► train.py ──► ckpt.pt ─┬─► train_grpo.py ──────────► grpo.pt
                                                │      ▲ reward: rules or a learned model
                                                │      │
                                                └─► make_preferences.py ──► preferences.jsonl
                                                                              │
                                                       train_reward_model.py ◄┤
                                                       train_dpo.py ◄─────────┘──► dpo.pt
```

## Quick start

```bash
cd SmallLM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Download and tokenize Shakespeare
python prepare.py --dataset shakespeare --tokenizer char

# 2. Pretrain (~0.8M params, ~6 min on a 4-core CPU)
python train.py --data data/shakespeare_char --preset tiny --max-iters 1500

# 3. Generate text
python sample.py --prompt "ROMEO:" --tokens 300
```

## Reinforcement learning

Pretraining teaches the model to imitate text. RL teaches it to produce text
that scores well on a goal you choose. Both methods below keep the model close
to a frozen copy of the pretrained model (the *reference*), so it doesn't forget
how to write while chasing the goal.

### GRPO: optimize a reward directly

```bash
python train_grpo.py --ckpt out/ckpt.pt --data data/shakespeare_char \
    --reward known_words --reward distinct_words --steps 150 --lr 3e-5
python sample.py --ckpt out/grpo.pt --prompt "ROMEO:"
```

For every prompt, the model writes a group of completions. Each completion is
scored and compared with the group average, and the model is pushed toward the
above-average ones. In a 150-step CPU run (~5 min), the reward rose from 1.55 to
1.75 out of 2.0.

Built-in rewards (`slm/rl/rewards.py`); repeat `--reward` to add several together:

| Reward | Scores |
|---|---|
| `known_words` | fraction of words that appear in the training text |
| `distinct_words` | fraction of words that aren't repeats |
| `keyword=love` | 1 if the word appears, else 0 |
| `model=out/reward_model.pt` | a learned reward model |

**Watch out for reward hacking.** With only `known_words`, the model soon learns
to write "the the the…", since every word is real. Adding `distinct_words`
closes that loophole. Read the samples, not just the reward curve.

### Preferences, reward models and DPO

Real RLHF starts from people comparing two answers. `make_preferences.py` fakes
that by using a rule-based reward as the judge. You can also write
`preferences.jsonl` yourself; each line is `{"prompt", "chosen", "rejected"}`.

```bash
# Make 800 (chosen, rejected) pairs
python make_preferences.py --ckpt out/ckpt.pt --data data/shakespeare_char \
    --reward known_words --reward distinct_words --pairs 800

# Option A: learn a reward model, then optimize it with GRPO (classic RLHF)
python train_reward_model.py --ckpt out/ckpt.pt --prefs data/preferences.jsonl
python train_grpo.py --ckpt out/ckpt.pt --data data/shakespeare_char --reward model=out/reward_model.pt

# Option B: DPO, which learns from the pairs directly with no reward model and no sampling
python train_dpo.py --ckpt out/ckpt.pt --prefs data/preferences.jsonl --lr 3e-5
python sample.py --ckpt out/dpo.pt --prompt "ROMEO:"
```

On the tiny model with 800 pairs, the reward model reached 57% validation accuracy
after 3 epochs, and DPO reached 65%. More pairs and a bigger model raise both.

## Code map

| File | What it does |
|---|---|
| **Scripts** | |
| `prepare.py` | Download or read text, tokenize it, write `train.bin`, `val.bin` and `tokenizer.json` |
| `train.py` | Pretrain with next-token prediction |
| `sample.py` | Generate text from any checkpoint |
| `train_grpo.py` | RL fine-tuning against a reward |
| `make_preferences.py` | Build preference pairs using a reward as the judge |
| `train_reward_model.py` | Learn a reward model from preference pairs |
| `train_dpo.py` | Fine-tune on preference pairs with DPO |
| **Library: `slm/`** | |
| `model.py` | The transformer: embeddings, causal self-attention, MLP blocks, generation |
| `tokenizer.py` | Character-level and GPT-2 BPE tokenizers |
| `data.py` | Load tokenized data and sample batches |
| `checkpoint.py` | Save and load a model together with its tokenizer |
| `utils.py` | Device selection and the learning-rate schedule |
| **RL: `slm/rl/`** | |
| `logprobs.py` | How likely a model is to produce given tokens; padding helpers |
| `rewards.py` | Rule-based rewards and how they're combined |
| `reward_model.py` | A transformer with a scoring head, trained with the Bradley-Terry loss |
| `grpo.py` | Group-relative advantages, the clipped objective and the KL penalty |
| `dpo.py` | The DPO loss |
| `preferences.py` | Read and write preference pairs |

All paths are relative to `SmallLM/`, and the scripts are run from there.

Start with `slm/model.py`, then `train.py`. For RL, read `slm/rl/grpo.py`, then
`train_grpo.py`.

## Model sizes

| Preset | Layers | Heads | Width | Context | Params (char vocab) | Where |
|---|---|---|---|---|---|---|
| `tiny` | 4 | 4 | 128 | 128 | ~0.8M | CPU |
| `small` | 6 | 6 | 384 | 256 | ~10M | GPU / Apple Silicon |
| `medium` | 8 | 8 | 512 | 512 | ~25M | GPU |

Any field can be overridden, e.g. `--n-layer 8 --n-embd 256`. The device is
picked automatically (CUDA, then Apple Silicon, then CPU); set `--device` to
choose. On CUDA, training runs in bfloat16; add `--compile` for extra speed.

## Other data

```bash
# TinyStories: simple English stories. Small models become coherent on this.
python prepare.py --dataset tinystories --tokenizer bpe
python train.py --data data/tinystories_bpe --preset small --max-iters 10000

# Your own text
python prepare.py --input my_corpus.txt --tokenizer bpe
```

The BPE tokenizer has 50,257 tokens, so for a small model the embedding table
is most of the parameters, and it needs more data and steps than char-level.

## Ideas for next steps

- Supervised fine-tuning (SFT) on prompt/response pairs before RL
- Modern components: RoPE, RMSNorm, SwiGLU, grouped-query attention
- A KV cache for faster generation (and faster GRPO sampling)
- A PPO trainer with a value head, to compare with GRPO
