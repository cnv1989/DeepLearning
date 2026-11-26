# 9. Reinforcement learning with GRPO

**Code:** [`train_grpo.py`](../train_grpo.py), [`slm/rl/grpo.py`](../slm/rl/grpo.py), [`slm/rl/rewards.py`](../slm/rl/rewards.py), [`slm/rl/logprobs.py`](../slm/rl/logprobs.py)

## What it is

SFT needs examples of the right answer. Often you can't write them, but you
**can recognize** a good answer: a maths answer can be checked, code can be run
against tests, a reply can be scored. Reinforcement learning (RL) trains on
that signal: the model tries, gets a score (a **reward**), and shifts toward
whatever scored well.

**GRPO** (Group Relative Policy Optimization) is the RL method behind recent
reasoning models such as DeepSeek-R1. It's popular because it's simple.

```bash
python train_grpo.py --ckpt out/sft/ckpt.pt --prompts data/tasks/train.jsonl \
    --reward exact_match --reward similarity --lr 3e-5
```

## The vocabulary of RL for LLMs

| Term | Here |
|---|---|
| **Policy** | the model being trained |
| **Action** | choosing the next token |
| **Episode** | writing one complete answer to a prompt |
| **Reward** | a score for the finished answer |
| **Reference model** | a frozen copy of the policy from before RL started |

## One GRPO step

```
for each of 8 prompts:
    1. sample a GROUP of 8 answers                "esor", "rose", "esro", "eros", ...
    2. score each one                             1.0,    0.5,    0.75,   0.5,   ...
    3. advantage = (score − group mean) / group std
                                                  +1.4,   −0.8,   +0.3,   −0.8,  ...
4. update the model: raise the probability of every token in positive-advantage
   answers and lower it in negative-advantage ones, in proportion to the advantage
```

### Why compare within a group? (`group_advantages`)

A reward of 0.6 means nothing by itself: is that good for this prompt?
Comparing each answer with other answers *to the same prompt* answers that.
Above average is good, below is bad, whatever the prompt's difficulty.

Older methods (PPO) train a second network, a "value model", to predict the
expected reward. GRPO replaces it with the group average: no extra model to
train, and less to go wrong.

If every answer in a group gets the same score, all advantages are 0 and the
group teaches nothing. This matters a lot in practice (see below).

### The loss (`grpo_loss`)

For each token, compare how likely it is now with how likely it was when sampled:

```
ratio = p_now(token) / p_when_sampled(token)
```

The objective is `ratio × advantage`, which is pushed up for good answers and down
for bad ones. Two safety rails are added:

**1. Clipping** (borrowed from PPO). The ratio is clipped to [0.8, 1.2], and
the smaller of the clipped and unclipped objectives is used. Once a token's
probability has moved 20% in the helpful direction, there's no more reward
for moving it further *on this batch*. That stops one batch from changing the
model too much, which matters because each batch is reused for
`--updates-per-step` gradient steps.

**2. KL penalty** (`kl_to_reference`). This measures how far the policy has
drifted from the reference model, and adds `kl_weight ×` that to the loss:

```
per-token KL estimate = exp(r) − r − 1,   with r = log p_ref − log p_policy
```

It's always ≥ 0, and 0 only when the two models agree. Without it, the model
can drift into strange text that happens to score well while forgetting how
to write.

### Variable-length answers (`completion_mask`)

Answers stop at EOS, so they have different lengths. Tokens after the first
EOS are padding the model never really chose, so they're masked out of the
loss and the KL.

## Rewards (`slm/rl/rewards.py`)

Every reward is a function `(prompts, completions, answers) -> scores`.
Repeat `--reward` to add several together.

| Reward | Kind | Scores |
|---|---|---|
| `exact_match` | verifiable | 1 if the answer is exactly right |
| `similarity` | verifiable | 0–1 for how close it is (`difflib`) |
| `known_words` | rule | fraction of words that are real (from the training text) |
| `distinct_words` | rule | fraction of words that aren't repeats |
| `keyword=love` | rule | 1 if the word appears |
| `model=PATH` | learned | a trained reward model ([next guide](10-reward-models-and-dpo.md)) |

**Verifiable rewards** (checking against a known answer) are the most reliable:
they can't be fooled. That's why maths and code are where RL for LLMs works
best.

### Reward hacking

RL optimizes exactly what you measure, not what you meant. With only
`known_words`, the model learns that "the the the the" scores a perfect 1.0.
Adding `distinct_words` closes that loophole. Every reward has loopholes, so
**always read the samples**, not just the reward curve.

## Prompts

| Flag | Prompts | Use with |
|---|---|---|
| `--prompts tasks.jsonl` | task prompts in the chat format, with answers | `exact_match`, `similarity` |
| `--data dir` | random 32-token snippets of training text | `known_words`, `distinct_words`, `keyword`, `model` |

## Results from test runs, and what they teach

**Open-ended text** (from the pretrained model, `known_words + distinct_words`,
100 steps): reward rose from 1.78 to 1.83 out of 2.0 (peaking at 1.86), with KL staying small (0.07).

**Tasks** (from an SFT model): no accuracy gain.

| Starting point | Start accuracy | After GRPO |
|---|---|---|
| SFT for 250 steps | 5.5% | 5.0% (60 steps) |
| SFT for 500 steps | 23.5% | 22.5% (400 steps, lr 3e-5) |
| SFT for 500 steps, lr 1e-4 | 23.5% | 21.5%; unstable, KL up to 3.4 |

Three lessons:

1. **The model must sometimes succeed.** At 5% accuracy, most groups of 8 have
   no correct answer, all advantages are ~0, and there's nothing to learn.
   `similarity` gives partial credit so near-misses can be compared, but it
   wasn't enough here.
2. **Keep the learning rate low.** At 1e-4, a *single* update moved the model
   1.4 nats per token away from the reference. At 3e-5 it moved about 100×
   less. Watch the `kl` column: steady growth is fine, sudden jumps are not.
3. **RL pays off when SFT can't do the job.** Here, longer SFT simply solved
   the tasks (100%), so RL had nothing left to fix. RL shines when you can
   check answers but can't easily write them, or when SFT plateaus.

## Settings

| Setting | Default | Notes |
|---|---|---|
| `--group-size` | 8 | more answers per prompt: better comparisons, more compute |
| `--prompts-per-step` | 8 | more prompts: smoother updates |
| `--lr` | 1e-5 | 3e-5 worked here; RL needs far smaller rates than pretraining |
| `--kl-weight` | 0.05 | higher keeps the model closer to the reference |
| `--clip` | 0.2 | how far one batch may move a token's probability |
| `--updates-per-step` | 2 | gradient steps per batch of samples |
| `--temperature` | 1.0 | exploration; lower makes groups too similar to compare |
| `--completion-len` | 64 | maximum answer length |

## What to watch

| Column | Healthy | Worrying |
|---|---|---|
| `reward` | trending up (it's noisy step to step) | flat: groups too similar, or lr too low |
| `kl` | small, growing slowly | jumps above ~1: lr too high |
| `clipped` | mostly 0–0.05 | high: updates too big |
| `sample` | sensible text | gibberish or repetition: reward hacking |

## Try it

```bash
python train_grpo.py --ckpt out/ckpt.pt --data data/shakespeare_char \
    --reward known_words --steps 100 --lr 3e-5
```

With `known_words` alone, watch the samples for reward hacking. Then add
`--reward distinct_words` and compare.
