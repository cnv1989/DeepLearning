"""A learned reward model.

Hand-written rules only capture simple things. A reward model learns what
"good" means from examples: pairs of completions where one was preferred.

Architecture: the GPT transformer, but instead of predicting the next token,
a single linear layer turns the hidden vector at the last token into a score.

Training uses the Bradley-Terry model of preferences:

    P(chosen beats rejected) = sigmoid(score(chosen) - score(rejected))

and minimizes -log of that probability, which pushes chosen scores up and
rejected scores down.
"""

from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.model import GPT, GPTConfig
from slm.rl.logprobs import pad_prompt_completion_pairs, truncate_pairs
from slm.tokenizer import tokenizer_from_dict


class RewardModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.backbone = GPT(config)
        self.score_head = nn.Linear(config.n_embd, 1)

    @classmethod
    def from_language_model(cls, language_model):
        """Start from a pretrained GPT, so the reward model already understands text."""
        reward_model = cls(language_model.config)
        reward_model.backbone.load_state_dict(language_model.state_dict())
        return reward_model

    def forward(self, tokens, lengths):
        """Score each sequence using the hidden state at its last real token.

        tokens:  (batch, seq_len), right-padded
        lengths: (batch,) number of real tokens in each row
        returns: (batch,) one score per sequence
        """
        hidden = self.backbone.hidden_states(tokens)
        last_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        last_hidden = hidden.gather(dim=1, index=last_index).squeeze(1)
        return self.score_head(last_hidden).squeeze(-1)

    def score_texts(self, tokenizer, prompts, completions, device):
        """Convenience wrapper: score plain-text (prompt, completion) pairs."""
        prompt_ids = [tokenizer.encode(p) for p in prompts]
        completion_ids = [tokenizer.encode(c) for c in completions]
        prompt_ids, completion_ids = truncate_pairs(prompt_ids, completion_ids, self.config.block_size)
        tokens, _, lengths = pad_prompt_completion_pairs(prompt_ids, completion_ids, device)
        return self(tokens, lengths)


def preference_loss(chosen_scores, rejected_scores):
    """Bradley-Terry loss: -log sigmoid(score_chosen - score_rejected)."""
    return -F.logsigmoid(chosen_scores - rejected_scores).mean()


def save_reward_model(path, model, tokenizer):
    torch.save(
        {
            "kind": "reward_model",
            "config": asdict(model.config),
            "model": model.state_dict(),
            "tokenizer": tokenizer.to_dict(),
        },
        path,
    )


def load_reward_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    assert checkpoint.get("kind") == "reward_model", f"{path} is not a reward model checkpoint"
    model = RewardModel(GPTConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, tokenizer_from_dict(checkpoint["tokenizer"])
