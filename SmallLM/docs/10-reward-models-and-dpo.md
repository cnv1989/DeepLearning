# 10. Learning from preferences: reward models and DPO

**Code:** [`make_preferences.py`](../make_preferences.py), [`train_reward_model.py`](../train_reward_model.py), [`train_dpo.py`](../train_dpo.py), [`slm/rl/reward_model.py`](../slm/rl/reward_model.py), [`slm/rl/dpo.py`](../slm/rl/dpo.py), [`slm/rl/preferences.py`](../slm/rl/preferences.py)

## What it is

For most things we want from a chatbot (helpful, honest, well written) there's
no answer key and no rule that captures "good". But people can easily say
which of **two** answers is better. Learning from those comparisons is
**RLHF** (reinforcement learning from human feedback), the step that made
ChatGPT-style assistants work.

There are two ways to use preference data, and this repo has both:

```
                         ┌─► train_reward_model.py ─► reward model ─► train_grpo.py --reward model=...
 preference pairs ───────┤        (classic RLHF: learn a scorer, then do RL against it)
 (chosen, rejected)      │
                         └─► train_dpo.py ─► tuned model
                                  (DPO: learn from the pairs directly, no RL loop)
```

## Preference data (`slm/rl/preferences.py`)

JSON lines, one comparison per line:

```json
{"prompt": "irgin voice\nThat babies lulls as",
 "chosen": " desss to you the you\nnot conturiot, and lookined Noves to.",
 "rejected": "e to beagand:\nSo bee der such armitus me poy."}
```

In real RLHF, people pick the winner. `make_preferences.py` simulates them:
it samples two completions per prompt from the model and lets a rule-based
reward be the judge. Pairs whose scores are too close to call
(`--min-gap`) are skipped, as people would mark them as ties. You can also
write the file yourself, with your own judgements.

```bash
python make_preferences.py --data data/shakespeare_char \
    --reward known_words --reward distinct_words --pairs 1000
```

## Option A: a reward model

### What it is (`RewardModel`)

A copy of the pretrained language model with its next-token output replaced
by a single number: a score for the whole text. It reads `prompt + answer`,
and a linear layer (`score_head`) turns the hidden state at the **last
token** into a score. That position has seen the entire text, thanks to
causal attention.

Starting from the pretrained model matters: it already understands
language, so it only has to learn what "good" means.

### How it learns: the Bradley-Terry model

We never know the "true" score of an answer, only which of two won. The
Bradley-Terry model links scores to wins:

```
P(chosen beats rejected) = sigmoid(score_chosen − score_rejected)
loss = −log sigmoid(score_chosen − score_rejected)        (preference_loss)
```

The loss pushes the chosen answer's score up and the rejected one's down. Only
the *difference* matters, so scores have no fixed scale, which is fine for
GRPO, since it compares answers within a group anyway.

```bash
python train_reward_model.py --ckpt out/ckpt.pt --prefs data/preferences.jsonl
python train_grpo.py --ckpt out/ckpt.pt --data data/shakespeare_char --reward model=out/reward_model.pt
```

The number to watch is **validation accuracy**: how often it ranks a held-out
pair correctly. 50% is chance. In test runs on the tiny model, it reached
57–68% from 400–800 pairs.

### The catch: reward hacking

A reward model is a *learned approximation* of what people like, and it has
blind spots. With enough optimization pressure, RL finds text the reward model
scores highly but people wouldn't: overly long answers, flattery, odd
phrasing. The KL penalty in GRPO is the main defence: it limits how far the
policy can go to exploit the scorer.

## Option B: DPO (Direct Preference Optimization)

### The idea

The DPO paper showed that the two steps of classic RLHF (fit a reward model,
then run RL with a KL penalty) can be collapsed into **one supervised loss** on
the preference pairs. No reward model, no sampling, no RL loop: just a
forward and backward pass over pairs, like SFT.

### The loss (`dpo_loss`)

For each pair, measure how much the policy has increased each answer's
log-probability compared with the frozen reference model:

```
chosen_gain   = log p_policy(chosen)   − log p_ref(chosen)
rejected_gain = log p_policy(rejected) − log p_ref(rejected)

loss = −log sigmoid( β · (chosen_gain − rejected_gain) )
```

It's the Bradley-Terry loss again, but the "score" is **β × gain**, the
model's **implicit reward**. The model is rewarded for raising chosen answers
*more than* rejected ones, relative to where it started.

* **β** (`--beta`, default 0.1) plays the role of the KL penalty. Small β lets
  the model move far from the reference; large β keeps it close.
* At the start, policy = reference, so every gain is 0 and the loss is exactly
  ln 2 ≈ 0.693 (a test checks this). Falling below that means it's learning.
* **Preference accuracy** is how often the implicit reward ranks the chosen
  answer higher. In a 2-epoch test run on 720 pairs, it reached 65%.

`completion_logprob_sums` computes `log p(answer | prompt)`: the sum of
the answer tokens' log-probabilities, with the prompt and padding masked out.

```bash
python train_dpo.py --ckpt out/ckpt.pt --prefs data/preferences.jsonl --lr 3e-5
```

## Reward model + RL vs DPO

| | Reward model + GRPO | DPO |
|---|---|---|
| Models in memory | policy, reference, reward model | policy, reference |
| Sampling during training | yes (slow) | no |
| Stability | needs tuning | usually stable |
| Learns from | its own fresh samples | a fixed set of pairs |
| Can improve beyond the data | yes, by exploring | limited to the pairs given |

DPO is the common choice when you have a good preference dataset and want
something simple. Reward-model RL is more work but can keep improving on new
samples. Many production pipelines use both.

## Common problems

* **Reward model stuck near 50%**: too few pairs, or the judge's preferences
  are inconsistent. Check a few pairs by hand.
* **DPO makes text worse while accuracy rises**: a known failure where it
  lowers the probability of *both* answers, the rejected one more. Watch the
  `chosen_reward` stat; if it goes very negative, raise β or lower the
  learning rate.
* **Pairs too easy**: if chosen and rejected differ obviously, the model
  learns little. Pairs that are close but clearly ordered teach the most.

## Try it

Write 20 pairs by hand in `data/my_prefs.jsonl` (prefer, say, answers that
mention love), then:

```bash
python train_dpo.py --ckpt out/ckpt.pt --prefs data/my_prefs.jsonl --epochs 5
python sample.py --ckpt out/dpo.pt --prompt "ROMEO:"
```
