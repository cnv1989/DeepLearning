"""Fine-tune a pretrained model on preference pairs with DPO.

Example:
    python train_dpo.py --ckpt out/ckpt.pt --prefs data/preferences.jsonl

No reward model and no sampling: the model learns straight from the
(chosen, rejected) pairs. See slm/rl/dpo.py for how the loss works.
Saves to out/dpo.pt; read it with sample.py.
"""

import argparse

import torch

from slm.checkpoint import load_checkpoint, save_checkpoint
from slm.rl.dpo import dpo_loss
from slm.rl.logprobs import completion_logprob_sums, pad_prompt_completion_pairs, truncate_pairs
from slm.rl.preferences import batches, load_preferences, split_preferences
from slm.utils import pick_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default="out/ckpt.pt")
    parser.add_argument("--prefs", default="data/preferences.jsonl")
    parser.add_argument("--out", default="out/dpo.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="how strongly to stay near the reference")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def logprobs_of_answers(model, tokenizer, pairs, device):
    """log p(chosen | prompt) and log p(rejected | prompt) for each pair."""
    prompts = [tokenizer.encode(p["prompt"]) for p in pairs]
    results = []
    for key in ("chosen", "rejected"):
        completions = [tokenizer.encode(p[key]) for p in pairs]
        trimmed_prompts, completions = truncate_pairs(prompts, completions, model.config.block_size)
        tokens, completion_mask, _ = pad_prompt_completion_pairs(trimmed_prompts, completions, device)
        results.append(completion_logprob_sums(model, tokens, completion_mask))
    return results


def dpo_batch(policy, reference, tokenizer, batch, device, beta):
    policy_chosen, policy_rejected = logprobs_of_answers(policy, tokenizer, batch, device)
    with torch.no_grad():
        reference_chosen, reference_rejected = logprobs_of_answers(reference, tokenizer, batch, device)
    return dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)


@torch.no_grad()
def evaluate(policy, reference, tokenizer, pairs, device, args):
    accuracies = [dpo_batch(policy, reference, tokenizer, batch, device, args.beta)[1]["accuracy"]
                  for batch in batches(pairs, args.batch_size, shuffle=False)]
    return sum(accuracies) / len(accuracies)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    policy, tokenizer, _ = load_checkpoint(args.ckpt, device, dropout=0.0)
    reference, _, _ = load_checkpoint(args.ckpt, device, dropout=0.0)
    reference.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)

    train_pairs, val_pairs = split_preferences(load_preferences(args.prefs))
    print(f"{len(train_pairs)} training pairs, {len(val_pairs)} validation pairs")

    for epoch in range(1, args.epochs + 1):
        for i, batch in enumerate(batches(train_pairs, args.batch_size), start=1):
            loss, stats = dpo_batch(policy, reference, tokenizer, batch, device, args.beta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            if i % 10 == 0:
                print(f"epoch {epoch} batch {i:3d} | loss {loss.item():.4f} | "
                      f"accuracy {stats['accuracy']:.2f} | margin {stats['margin']:.3f}")
        val_accuracy = evaluate(policy, reference, tokenizer, val_pairs, device, args)
        print(f"epoch {epoch} done | val preference accuracy {val_accuracy:.2%}")

    save_checkpoint(args.out, policy, tokenizer, rl_method="dpo")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
