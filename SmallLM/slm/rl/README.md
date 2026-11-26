# `slm/rl/`: reinforcement learning and preferences

Everything that tunes a model toward a goal rather than toward imitating text.

| File | What it does | Guide |
|---|---|---|
| [`logprobs.py`](logprobs.py) | Log-probabilities of tokens under a model; completion masks; padding | used by all of the below |
| [`rewards.py`](rewards.py) | Verifiable, rule-based and learned rewards, and combining them | [GRPO](../../docs/09-reinforcement-learning-grpo.md) |
| [`grpo.py`](grpo.py) | Group-relative advantages, the clipped objective, the KL penalty | [GRPO](../../docs/09-reinforcement-learning-grpo.md) |
| [`preferences.py`](preferences.py) | Reading, writing and splitting (chosen, rejected) pairs | [Preferences](../../docs/10-reward-models-and-dpo.md) |
| [`reward_model.py`](reward_model.py) | A transformer that scores text, trained with the Bradley-Terry loss | [Preferences](../../docs/10-reward-models-and-dpo.md) |
| [`dpo.py`](dpo.py) | The Direct Preference Optimization loss | [Preferences](../../docs/10-reward-models-and-dpo.md) |

```
 SFT model ─► train_grpo.py ◄── reward: rewards.py (rules, exact answers) or reward_model.py
          └─► train_dpo.py  ◄── preference pairs (preferences.py)
```
