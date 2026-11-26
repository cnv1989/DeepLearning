"""Fine-tune a pretrained model with GRPO against a reward.

Example (tiny Shakespeare model, CPU, ~10 minutes):
    python train_grpo.py --ckpt out/ckpt.pt --data data/shakespeare_char \
        --reward known_words --reward distinct_words --steps 200

Each step:
    1. take a batch of prompts (random snippets of training text)
    2. sample `group-size` completions for every prompt
    3. score the completions with the reward
    4. update the model with the GRPO loss (see slm/rl/grpo.py)

Saves the fine-tuned model to out/grpo.pt; read it with sample.py.
"""

import argparse
import time

import torch

from slm.checkpoint import load_checkpoint, save_checkpoint
from slm.data import load_tokens, random_windows
from slm.rl.grpo import group_advantages, grpo_loss
from slm.rl.logprobs import completion_logprobs
from slm.rl.rewards import build_reward
from slm.utils import pick_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default="out/ckpt.pt", help="pretrained model to start from")
    parser.add_argument("--data", required=True, help="dataset directory; prompts are drawn from it")
    parser.add_argument("--out", default="out/grpo.pt")
    parser.add_argument("--reward", action="append", required=True,
                        help="reward to maximize; repeat to add several (see slm/rl/rewards.py)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8, help="completions sampled per prompt")
    parser.add_argument("--prompt-len", type=int, default=32, help="prompt length in tokens")
    parser.add_argument("--completion-len", type=int, default=64, help="completion length in tokens")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--updates-per-step", type=int, default=2,
                        help="gradient steps on each batch of samples")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=0.2, help="PPO clipping range")
    parser.add_argument("--kl-weight", type=float, default=0.05, help="strength of the pull toward the reference")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    # The policy is the model we train. The reference is a frozen copy of where it started.
    policy, tokenizer, _ = load_checkpoint(args.ckpt, device, dropout=0.0)
    reference, _, _ = load_checkpoint(args.ckpt, device, dropout=0.0)
    reference.requires_grad_(False)
    policy.eval()  # dropout is off; eval mode just makes that explicit
    reference.eval()

    total_len = args.prompt_len + args.completion_len
    assert total_len <= policy.config.block_size, (
        f"prompt-len + completion-len = {total_len} exceeds the model's block_size {policy.config.block_size}"
    )

    train_tokens = load_tokens(args.data, "train", tokenizer.vocab_size)
    reward_fn = build_reward(args.reward, args.data, tokenizer, device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)

    print(f"device {device} | rewards {args.reward} | "
          f"{args.prompts_per_step} prompts x {args.group_size} completions per step")
    start_time = time.time()

    for step in range(1, args.steps + 1):
        # ----- 1 & 2: sample a group of completions for each prompt -----
        prompts = random_windows(train_tokens, args.prompts_per_step, args.prompt_len, device)
        prompts = prompts.repeat_interleave(args.group_size, dim=0)  # each prompt G times in a row
        with torch.no_grad():
            sequences = policy.generate(prompts, args.completion_len, temperature=args.temperature)
            old_logprobs = completion_logprobs(policy, sequences, args.prompt_len)
            reference_logprobs = completion_logprobs(reference, sequences, args.prompt_len)

        # ----- 3: score them -----
        prompt_texts = [tokenizer.decode(s[: args.prompt_len].tolist()) for s in sequences]
        completion_texts = [tokenizer.decode(s[args.prompt_len:].tolist()) for s in sequences]
        rewards = torch.tensor(reward_fn(prompt_texts, completion_texts), device=device)
        advantages = group_advantages(rewards, args.group_size)

        # ----- 4: push the model toward the better completions -----
        for _ in range(args.updates_per_step):
            logprobs = completion_logprobs(policy, sequences, args.prompt_len)
            loss, stats = grpo_loss(logprobs, old_logprobs, reference_logprobs, advantages,
                                    clip=args.clip, kl_weight=args.kl_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

        if step % args.log_interval == 0 or step == 1:
            print(f"step {step:4d} | reward {rewards.mean().item():.3f} | kl {stats['kl']:.4f} "
                  f"| clipped {stats['clipped_fraction']:.2f} | {time.time() - start_time:.0f}s")
            print(f"    sample: {completion_texts[0]!r}")

    save_checkpoint(args.out, policy, tokenizer, rl_method="grpo", rewards=args.reward)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
