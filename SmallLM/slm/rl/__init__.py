"""Reinforcement learning for the language model.

After pretraining, the model imitates its training text. RL instead tunes it
to produce text that scores well on a reward. There are two families here:

1. Reward-driven RL (GRPO)
   Sample completions -> score each with a reward -> push the model toward
   the higher-scoring ones. The reward can be a hand-written rule
   (rewards.py) or a learned model (reward_model.py).

2. Preference learning (DPO)
   Given pairs of (chosen, rejected) completions, make the model prefer the
   chosen ones directly. No reward model and no sampling during training.

Both keep the model close to a frozen copy of where it started (the
"reference" model), so it doesn't forget how to write while chasing reward.

Modules
-------
logprobs      Log-probabilities of tokens under a model (used by everything).
rewards       Rule-based reward functions and a way to combine them.
reward_model  A transformer that learns to score text from preference pairs.
preferences   Reading and writing preference pairs as JSON lines.
grpo          Group Relative Policy Optimization loss.
dpo           Direct Preference Optimization loss.
"""
