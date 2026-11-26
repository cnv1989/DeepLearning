"""Generate preference pairs automatically, for the reward model and DPO.

In real RLHF, people compare two answers and pick the better one. Here a
rule-based reward plays the role of the person:

    for each prompt: sample two completions, score both,
    and record the higher-scoring one as "chosen".

Example:
    python make_preferences.py --ckpt out/ckpt.pt --data data/shakespeare_char \
        --reward known_words --reward distinct_words --pairs 1000

Writes JSON lines to data/preferences.jsonl. You can also write that file
yourself with human judgements; see slm/rl/preferences.py for the format.
"""

import argparse

import torch

from slm.checkpoint import load_checkpoint
from slm.data import load_tokens, random_windows
from slm.rl.preferences import save_preferences
from slm.rl.rewards import build_reward
from slm.utils import pick_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default="out/ckpt.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="data/preferences.jsonl")
    parser.add_argument("--reward", action="append", required=True, help="judge used to pick the winner")
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64, help="prompts sampled at a time")
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--completion-len", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-gap", type=float, default=0.05,
                        help="skip pairs whose scores are closer than this (too close to call)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    model, tokenizer, _ = load_checkpoint(args.ckpt, device, dropout=0.0)
    model.eval()
    train_tokens = load_tokens(args.data, "train", tokenizer.vocab_size)
    judge = build_reward(args.reward, args.data, tokenizer, device)

    pairs = []
    while len(pairs) < args.pairs:
        prompts = random_windows(train_tokens, args.batch_size, args.prompt_len, device)
        # Two completions per prompt: rows 0..B-1 are "A", rows B..2B-1 are "B".
        sequences = model.generate(torch.cat([prompts, prompts]), args.completion_len,
                                   temperature=args.temperature)

        prompt_texts = [tokenizer.decode(p.tolist()) for p in prompts]
        completions = [tokenizer.decode(s[args.prompt_len:].tolist()) for s in sequences]
        answers_a, answers_b = completions[: args.batch_size], completions[args.batch_size:]
        scores_a, scores_b = judge(prompt_texts, answers_a), judge(prompt_texts, answers_b)

        for prompt, a, b, score_a, score_b in zip(prompt_texts, answers_a, answers_b, scores_a, scores_b):
            if abs(score_a - score_b) < args.min_gap:
                continue
            chosen, rejected = (a, b) if score_a > score_b else (b, a)
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        print(f"{min(len(pairs), args.pairs)} / {args.pairs} pairs")

    save_preferences(args.out, pairs[: args.pairs])
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
