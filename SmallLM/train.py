"""Step 2: pretrain the model to predict the next token.

Examples:
    python train.py --config configs/tiny.json --data data/shakespeare_char      (CPU, ~6 min)
    python train.py --config configs/small.json --data data/tinystories_bpe      (one GPU)
    torchrun --nproc_per_node 4 train.py --config configs/medium.json --data ... (4 GPUs)

Any setting below can be given in the JSON config or as a flag (--max-iters 3000).
Saves the best model to <out>/ckpt.pt, the latest state to <out>/last.pt
(continue with --resume), and metrics to <out>/metrics.jsonl.
"""

import os

import torch

from slm.config import load_settings
from slm.data import get_batch, load_tokenizer, load_tokens
from slm.distributed import cleanup_distributed, setup_distributed
from slm.metrics import MetricsLogger
from slm.model import GPT, GPTConfig
from slm.trainer import Trainer
from slm.utils import pick_device

DEFAULTS = dict(
    data=None,  # directory made by prepare.py
    out="out",
    # model shape
    n_layer=4, n_head=4, n_kv_head=0, n_embd=128, block_size=128, dropout=0.0,
    # optimization
    batch_size=32, grad_accum=1, max_iters=5000, lr=1e-3, min_lr=1e-4, warmup_iters=100,
    weight_decay=0.1, grad_clip=1.0,
    # evaluation and logging
    eval_interval=250, eval_iters=50, log_interval=10, wandb=False,
    # hardware
    device="auto", dtype="auto", compile=False, activation_checkpointing=False,
    resume=False, seed=1337,
)


def main():
    settings = load_settings(DEFAULTS, __doc__)
    assert settings.data, "--data is required"
    context = setup_distributed(pick_device(settings.device))
    torch.manual_seed(settings.seed + context.rank)  # different batches on each GPU
    os.makedirs(settings.out, exist_ok=True)

    tokenizer = load_tokenizer(settings.data)
    datasets = {split: load_tokens(settings.data, split, tokenizer.vocab_size) for split in ("train", "val")}

    def batch(split):
        return get_batch(datasets[split], settings.batch_size, settings.block_size, context.device)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=settings.block_size, n_layer=settings.n_layer,
        n_head=settings.n_head, n_kv_head=settings.n_kv_head, n_embd=settings.n_embd,
        dropout=settings.dropout,
    )
    model = GPT(config)
    model.activation_checkpointing = settings.activation_checkpointing

    logger = MetricsLogger(settings.out, vars(settings), settings.wandb, enabled=context.is_main)
    trainer = Trainer(model, tokenizer, settings, batch, context, logger)
    trainer.print(f"device {context.device} x {context.world_size} | dtype {trainer.dtype} | "
                  f"{model.num_params() / 1e6:.2f}M parameters | vocab {tokenizer.vocab_size}")
    trainer.train()

    logger.close()
    cleanup_distributed(context)


if __name__ == "__main__":
    main()
