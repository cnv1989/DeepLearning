"""Measure how good a checkpoint is.

    python evaluate.py --ckpt out/ckpt.pt --data data/shakespeare_char
    python evaluate.py --ckpt out/sft/ckpt.pt --tasks data/tasks/test.jsonl

Two kinds of evaluation:

  Perplexity (--data)
      How surprised the model is by held-out text: exp(average loss) over the
      whole validation set, read in order. Lower is better. A perplexity of 5
      means the model is, on average, as unsure as choosing among 5 tokens.

  Task accuracy (--tasks)
      For prompts with known answers: generate a reply (greedily, so results
      are repeatable) and count exact matches, per task and overall.
"""

import argparse
import math
from collections import defaultdict

import torch

from slm.checkpoint import load_checkpoint
from slm.data import load_jsonl, load_tokens
from slm.sft import generate_reply
from slm.utils import pick_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default="out/ckpt.pt")
    parser.add_argument("--data", help="dataset directory: report validation perplexity")
    parser.add_argument("--tasks", help="JSON lines with prompt/answer: report accuracy")
    parser.add_argument("--max-tokens", type=int, default=500_000, help="cap on validation tokens read")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--show", type=int, default=5, help="print this many example replies")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.no_grad()
def perplexity(model, tokens, block_size, batch_size, max_tokens, device):
    """exp(mean loss) over consecutive, non-overlapping windows of the token stream."""
    n_windows = min(len(tokens) - 1, max_tokens) // block_size
    total_loss, total_tokens = 0.0, 0
    for start in range(0, n_windows, batch_size):
        rows = range(start, min(start + batch_size, n_windows))
        windows = torch.stack([
            torch.from_numpy(tokens[r * block_size : r * block_size + block_size + 1].astype("int64"))
            for r in rows
        ]).to(device)
        _, loss = model(windows[:, :-1], windows[:, 1:])
        n = windows[:, 1:].numel()
        total_loss += loss.item() * n
        total_tokens += n
    mean_loss = total_loss / total_tokens
    return mean_loss, math.exp(mean_loss), total_tokens


def task_accuracy(model, tokenizer, examples, max_new_tokens, show):
    correct_by_task, total_by_task = defaultdict(int), defaultdict(int)
    for i, example in enumerate(examples):
        reply = generate_reply(model, tokenizer, [{"role": "user", "content": example["prompt"]}],
                               max_new_tokens=max_new_tokens)
        is_correct = reply.strip() == example["answer"].strip()
        task = example.get("task", "all")
        correct_by_task[task] += is_correct
        total_by_task[task] += 1
        if i < show:
            mark = "✓" if is_correct else "✗"
            print(f"  {mark} {example['prompt']!r} -> {reply!r} (expected {example['answer']!r})")
    return correct_by_task, total_by_task


def main():
    args = parse_args()
    device = pick_device(args.device)
    model, tokenizer, _ = load_checkpoint(args.ckpt, device, dropout=0.0)
    model.eval()
    assert args.data or args.tasks, "give --data and/or --tasks"

    if args.data:
        tokens = load_tokens(args.data, "val", tokenizer.vocab_size)
        loss, ppl, n = perplexity(model, tokens, model.config.block_size, args.batch_size, args.max_tokens, device)
        print(f"validation: loss {loss:.4f} | perplexity {ppl:.2f} | {n:,} tokens")

    if args.tasks:
        examples = load_jsonl(args.tasks)
        correct, total = task_accuracy(model, tokenizer, examples, args.max_new_tokens, args.show)
        for task in sorted(total):
            print(f"  {task:>14}: {correct[task] / total[task]:6.1%}  ({correct[task]}/{total[task]})")
        print(f"accuracy: {sum(correct.values()) / len(examples):.1%} on {len(examples)} examples")


if __name__ == "__main__":
    main()
