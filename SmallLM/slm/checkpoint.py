"""Saving and loading model checkpoints.

A checkpoint is a single .pt file holding everything needed to use the model:
its config, its weights, and its tokenizer. Every training script (pretraining,
GRPO, DPO) writes this same format, so sample.py works with all of them.
"""

from dataclasses import asdict

import torch

from slm.model import GPT, GPTConfig
from slm.tokenizer import tokenizer_from_dict


def save_checkpoint(path, model, tokenizer, **extra):
    """Save a model. Anything in `extra` (optimizer state, step...) is stored too."""
    torch.save(
        {
            "config": asdict(model.config),
            "model": model.state_dict(),
            "tokenizer": tokenizer.to_dict(),
            **extra,
        },
        path,
    )


def load_checkpoint(path, device="cpu", dropout=None):
    """Load a model saved with save_checkpoint.

    Pass dropout=0.0 for fine-tuning with RL or for inference.
    Returns (model, tokenizer, checkpoint_dict).
    """
    checkpoint = torch.load(path, map_location=device)
    config = GPTConfig(**checkpoint["config"])
    if dropout is not None:
        config.dropout = dropout
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    return model, tokenizer, checkpoint
