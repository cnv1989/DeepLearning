"""GRPO: Group Relative Policy Optimization.

The idea, for each prompt:

  1. Sample a group of G completions from the current model (the "policy").
  2. Score each one with the reward function.
  3. Compare each completion to the rest of its group:
         advantage = (reward - group mean) / group std
     Better-than-average completions get a positive advantage,
     worse-than-average ones a negative advantage.
  4. Raise the probability of tokens in positive-advantage completions
     and lower it for negative ones.

Using the group average as the baseline is what makes GRPO simple: unlike
PPO it needs no separate value network to estimate "how good is average".

Two safety rails keep training stable:

  * Clipping (from PPO): the probability ratio new/old is clipped to
    [1 - clip, 1 + clip], so one update can't move the model too far.
  * KL penalty: a term that grows as the model drifts away from the frozen
    reference model, so it keeps writing like itself while chasing reward.
"""

import torch


def group_advantages(rewards, group_size, eps=1e-6):
    """Normalize rewards within each group of completions for the same prompt.

    rewards: (num_prompts * group_size,), ordered so each group is contiguous
    returns: same shape; mean 0 and std 1 within each group
    """
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    return ((grouped - mean) / (std + eps)).view(-1)


def kl_to_reference(policy_logprobs, reference_logprobs):
    """Per-token estimate of KL(policy || reference).

    Uses the "k3" estimator  exp(r) - r - 1  with  r = log p_ref - log p_policy.
    It is always >= 0 and is zero exactly when the two models agree.
    """
    log_ratio = reference_logprobs - policy_logprobs
    return torch.exp(log_ratio) - log_ratio - 1


def grpo_loss(policy_logprobs, old_logprobs, reference_logprobs, advantages, clip=0.2, kl_weight=0.05):
    """The GRPO objective (to minimize).

    policy_logprobs:    (batch, completion_len) under the model being trained (with gradients)
    old_logprobs:       (batch, completion_len) under the model that sampled the data
    reference_logprobs: (batch, completion_len) under the frozen reference model
    advantages:         (batch,) one number per completion, shared by all its tokens

    Returns (loss, stats) where stats is a dict of numbers worth logging.
    """
    advantages = advantages.unsqueeze(1)  # broadcast over tokens

    # How much more (or less) likely each token is now than when it was sampled.
    ratio = torch.exp(policy_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1 - clip, 1 + clip)

    # PPO's pessimistic choice: take the smaller of the clipped and unclipped objective.
    objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    policy_loss = -objective.mean()

    kl = kl_to_reference(policy_logprobs, reference_logprobs).mean()
    loss = policy_loss + kl_weight * kl

    stats = {
        "policy_loss": policy_loss.item(),
        "kl": kl.item(),
        "clipped_fraction": ((ratio - 1).abs() > clip).float().mean().item(),
    }
    return loss, stats
