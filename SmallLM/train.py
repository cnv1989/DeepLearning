"""Step 2: pretrain the model to predict the next token.

Example (CPU, ~6 minutes):
    python train.py --data data/shakespeare_char --preset tiny --max-iters 1500

Saves the best checkpoint (lowest validation loss) to out/ckpt.pt.
"""

import argparse
import math
import os
import time
from contextlib import nullcontext

import torch

from slm.checkpoint import load_checkpoint, save_checkpoint
from slm.data import get_batch, load_tokenizer, load_tokens
from slm.model import GPT, GPTConfig
from slm.utils import cosine_lr, pick_device, set_learning_rate

MODEL_PRESETS = {
    # ~0.8M params: trains in minutes on a laptop CPU.
    "tiny": dict(n_layer=4, n_head=4, n_embd=128, block_size=128, batch_size=32),
    # ~10M params: for a GPU or Apple Silicon.
    "small": dict(n_layer=6, n_head=6, n_embd=384, block_size=256, batch_size=64),
    # ~25M params: for TinyStories-sized data on a GPU.
    "medium": dict(n_layer=8, n_head=8, n_embd=512, block_size=512, batch_size=32),
}
PRESET_FIELDS = ["n_layer", "n_head", "n_embd", "block_size", "batch_size"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="directory made by prepare.py")
    parser.add_argument("--out", default="out", help="where to save ckpt.pt")
    parser.add_argument("--preset", choices=MODEL_PRESETS, default="tiny")
    for field in PRESET_FIELDS:
        parser.add_argument(f"--{field.replace('_', '-')}", type=int, help="override the preset")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=1, help="micro-batches per optimizer step")
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--compile", action="store_true", help="use torch.compile (GPU speed-up)")
    parser.add_argument("--resume", action="store_true", help="continue from <out>/ckpt.pt")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    # Fill in anything not given on the command line from the preset.
    for field, value in MODEL_PRESETS[args.preset].items():
        if getattr(args, field) is None:
            setattr(args, field, value)
    return args


def mixed_precision(device):
    """On CUDA, run in bfloat16 for speed. Elsewhere, use full precision."""
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def estimate_loss(model, datasets, args, device):
    """Average loss over a few random batches from each split."""
    model.eval()
    losses = {}
    for split, tokens in datasets.items():
        total = 0.0
        for _ in range(args.eval_iters):
            inputs, targets = get_batch(tokens, args.batch_size, args.block_size, device)
            with mixed_precision(device):
                _, loss = model(inputs, targets)
            total += loss.item()
        losses[split] = total / args.eval_iters
    model.train()
    return losses


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    checkpoint_path = os.path.join(args.out, "ckpt.pt")

    # ----- data -----
    tokenizer = load_tokenizer(args.data)
    datasets = {split: load_tokens(args.data, split, tokenizer.vocab_size) for split in ("train", "val")}

    # ----- model and optimizer -----
    start_iter, best_val_loss = 0, float("inf")
    if args.resume and os.path.exists(checkpoint_path):
        model, _, checkpoint = load_checkpoint(checkpoint_path, device)
        optimizer = model.configure_optimizer(args.lr, args.weight_decay)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter, best_val_loss = checkpoint["iter"] + 1, checkpoint["best_val_loss"]
        print(f"resuming from iteration {checkpoint['iter']}")
    else:
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
            n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
        )
        model = GPT(config).to(device)
        optimizer = model.configure_optimizer(args.lr, args.weight_decay)

    print(f"device {device} | {model.num_params() / 1e6:.2f}M parameters | vocab {tokenizer.vocab_size}")
    compiled_model = torch.compile(model) if args.compile else model

    # ----- training loop -----
    start_time = time.time()
    for iteration in range(start_iter, args.max_iters + 1):
        lr = cosine_lr(iteration, args.lr, args.min_lr, args.warmup_iters, args.max_iters)
        set_learning_rate(optimizer, lr)

        # Every so often: measure progress and save if this is the best model so far.
        if iteration % args.eval_interval == 0 or iteration == args.max_iters:
            losses = estimate_loss(compiled_model, datasets, args, device)
            print(f"iter {iteration:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} "
                  f"| val perplexity {math.exp(losses['val']):6.2f} | {time.time() - start_time:.0f}s")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(checkpoint_path, model, tokenizer, optimizer=optimizer.state_dict(),
                                iter=iteration, best_val_loss=best_val_loss)
            if iteration == args.max_iters:
                break

        # One optimizer step, possibly accumulated over several micro-batches.
        for _ in range(args.grad_accum):
            inputs, targets = get_batch(datasets["train"], args.batch_size, args.block_size, device)
            with mixed_precision(device):
                _, loss = compiled_model(inputs, targets)
            (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    print(f"done. best val loss {best_val_loss:.4f}, saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
