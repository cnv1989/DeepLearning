"""Step 3: supervised fine-tuning (SFT) on prompt/response examples.

Example:
    python make_task_data.py --data data/shakespeare_char
    python train_sft.py --ckpt out/ckpt.pt --train-file data/tasks/train.jsonl

Starts from a pretrained checkpoint and trains it to reply in the chat format
described in slm/sft.py. Only the assistant's replies are trained on.
Saves to out/sft/ckpt.pt. Works with torchrun for multiple GPUs, like train.py.
"""

import os

import torch

from slm.checkpoint import load_checkpoint
from slm.config import load_settings
from slm.data import load_jsonl
from slm.distributed import cleanup_distributed, setup_distributed
from slm.metrics import MetricsLogger
from slm.sft import SFTDataset
from slm.trainer import Trainer
from slm.utils import pick_device

DEFAULTS = dict(
    ckpt="out/ckpt.pt",  # pretrained model to start from
    train_file=None,  # JSON lines of {"prompt", "response"} or {"messages"}
    val_file=None,  # optional; otherwise 5% of train_file is held out
    out="out/sft",
    batch_size=32, grad_accum=1, max_iters=1000, lr=3e-4, min_lr=3e-5, warmup_iters=50,
    weight_decay=0.0, grad_clip=1.0,
    eval_interval=100, eval_iters=20, log_interval=10, wandb=False,
    device="auto", dtype="auto", compile=False, activation_checkpointing=False,
    resume=False, seed=1337,
)


def main():
    settings = load_settings(DEFAULTS, __doc__)
    assert settings.train_file, "--train-file is required"
    context = setup_distributed(pick_device(settings.device))
    torch.manual_seed(settings.seed + context.rank)
    os.makedirs(settings.out, exist_ok=True)

    model, tokenizer, _ = load_checkpoint(settings.ckpt, context.device)
    model.activation_checkpointing = settings.activation_checkpointing
    block_size = model.config.block_size

    train_examples = load_jsonl(settings.train_file)
    if settings.val_file:
        val_examples = load_jsonl(settings.val_file)
    else:
        n_val = max(1, len(train_examples) // 20)
        train_examples, val_examples = train_examples[n_val:], train_examples[:n_val]
    datasets = {
        "train": SFTDataset(train_examples, tokenizer, block_size),
        "val": SFTDataset(val_examples, tokenizer, block_size),
    }

    def batch(split):
        return datasets[split].get_batch(settings.batch_size, context.device)

    logger = MetricsLogger(settings.out, vars(settings), settings.wandb, enabled=context.is_main)
    trainer = Trainer(model, tokenizer, settings, batch, context, logger)
    trainer.print(f"{len(datasets['train'])} training and {len(datasets['val'])} validation examples")
    trainer.train()

    logger.close()
    cleanup_distributed(context)


if __name__ == "__main__":
    main()
