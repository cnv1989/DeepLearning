"""Small helpers shared by the training scripts."""

import math
from contextlib import nullcontext

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


def pick_dtype(name, device):
    """Number format for the forward/backward pass.

    'auto' uses bfloat16 on GPUs that support it, float16 on older GPUs,
    and float32 everywhere else. Weights are always stored in float32.
    """
    if name != "auto":
        return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def autocast(device, dtype):
    """Run the enclosed code in lower precision where it is safe to (mixed precision)."""
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type=device.split(":")[0], dtype=dtype)


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
