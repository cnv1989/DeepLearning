"""Train a reward model on preference pairs.

Example:
    python train_reward_model.py --ckpt out/ckpt.pt --prefs data/preferences.jsonl

The reward model starts as a copy of the pretrained language model with a
scoring head on top (see slm/rl/reward_model.py). Once trained, use it as a
GRPO reward:
    python train_grpo.py ... --reward model=out/reward_model.pt
"""

import argparse

import torch

from slm.checkpoint import load_checkpoint
from slm.rl.logprobs import pad_prompt_completion_pairs, truncate_pairs
from slm.rl.preferences import batches, load_preferences, split_preferences
from slm.rl.reward_model import RewardModel, preference_loss, save_reward_model
from slm.utils import pick_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default="out/ckpt.pt", help="pretrained model to initialize from")
    parser.add_argument("--prefs", default="data/preferences.jsonl")
    parser.add_argument("--out", default="out/reward_model.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def score_pairs(model, tokenizer, pairs, device):
    """Return (chosen_scores, rejected_scores) for a batch of preference pairs."""
    prompts = [tokenizer.encode(p["prompt"]) for p in pairs]
    scores = []
    for key in ("chosen", "rejected"):
        completions = [tokenizer.encode(p[key]) for p in pairs]
        trimmed_prompts, completions = truncate_pairs(prompts, completions, model.config.block_size)
        tokens, _, lengths = pad_prompt_completion_pairs(trimmed_prompts, completions, device)
        scores.append(model(tokens, lengths))
    return scores


@torch.no_grad()
def accuracy(model, tokenizer, pairs, device, batch_size):
    """Fraction of pairs where the chosen answer gets the higher score."""
    model.eval()
    correct = 0
    for batch in batches(pairs, batch_size, shuffle=False):
        chosen, rejected = score_pairs(model, tokenizer, batch, device)
        correct += (chosen > rejected).sum().item()
    model.train()
    return correct / len(pairs)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    language_model, tokenizer, _ = load_checkpoint(args.ckpt, device, dropout=0.0)
    model = RewardModel.from_language_model(language_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_pairs, val_pairs = split_preferences(load_preferences(args.prefs))
    print(f"{len(train_pairs)} training pairs, {len(val_pairs)} validation pairs")
    print(f"before training: val accuracy {accuracy(model, tokenizer, val_pairs, device, args.batch_size):.2%}")

    for epoch in range(1, args.epochs + 1):
        total_loss, num_batches = 0.0, 0
        for batch in batches(train_pairs, args.batch_size):
            chosen, rejected = score_pairs(model, tokenizer, batch, device)
            loss = preference_loss(chosen, rejected)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        val_acc = accuracy(model, tokenizer, val_pairs, device, args.batch_size)
        print(f"epoch {epoch} | train loss {total_loss / num_batches:.4f} | val accuracy {val_acc:.2%}")

    save_reward_model(args.out, model, tokenizer)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
