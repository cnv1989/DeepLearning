"""DPO: Direct Preference Optimization.

Classic RLHF trains a reward model, then runs RL against it. DPO shows the two
steps can be merged into one supervised loss on preference pairs.

For each (prompt, chosen, rejected) it compares how much the policy has
increased the log-probability of each answer relative to the reference model:

    chosen_gain   = log p_policy(chosen)   - log p_ref(chosen)
    rejected_gain = log p_policy(rejected) - log p_ref(rejected)

    loss = -log sigmoid( beta * (chosen_gain - rejected_gain) )

So the model is rewarded for raising chosen answers more than rejected ones.
`beta` controls how far it may move from the reference: a small beta allows
big changes; a large beta keeps it close.

beta * gain is called the "implicit reward". If it ranks the chosen answer
higher, the model has learned that preference.
"""

import torch.nn.functional as F


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    """All inputs are (batch,) total log-probabilities of each completion.

    Returns (loss, stats).
    """
    chosen_reward = beta * (policy_chosen - reference_chosen)
    rejected_reward = beta * (policy_rejected - reference_rejected)
    margin = chosen_reward - rejected_reward

    loss = -F.logsigmoid(margin).mean()
    stats = {
        "accuracy": (margin > 0).float().mean().item(),
        "margin": margin.mean().item(),
        "chosen_reward": chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
    }
    return loss, stats
