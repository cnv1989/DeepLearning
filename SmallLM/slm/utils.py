"""Small helpers shared by the training scripts."""

import math

import torch


def pick_device(name="auto"):
    """'auto' picks the fastest available device: CUDA GPU, then Apple Silicon, then CPU."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cosine_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    """Learning rate with linear warmup followed by cosine decay.

    lr
    max ┤   ╭──╮
        │  ╱    ╲_
        │ ╱       ╲__
    min ┤╱           ╲____
        └─┬───────────────── step
        warmup
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def set_learning_rate(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr
