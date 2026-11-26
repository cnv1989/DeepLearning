"""Fine-tune a model with GRPO (reinforcement learning against a reward).

Two kinds of prompts:

  Task prompts with known answers (verifiable rewards), usually after SFT:
    python train_grpo.py --ckpt out/sft/ckpt.pt --prompts data/tasks/train.jsonl \
        --reward exact_match --reward similarity --steps 200

  Open-ended text: random snippets of the training data as prompts:
    python train_grpo.py --ckpt out/ckpt.pt --data data/shakespeare_char \
        --reward known_words --reward distinct_words --steps 150

Each step:
    1. take a batch of prompts
    2. sample `group-size` completions for every prompt
    3. score the completions with the reward
    4. update the model with the GRPO loss (see slm/rl/grpo.py)

Saves the fine-tuned model to out/grpo.pt; read it with sample.py or evaluate.py.
"""

import argparse
import random
import time
from dataclasses import dataclass

import torch

from slm.checkpoint import load_checkpoint, save_checkpoint
from slm.data import load_jsonl, load_tokens, random_windows
from slm.rl.grpo import group_advantages, grpo_loss
from slm.rl.logprobs import completion_logprobs, completion_mask
from slm.rl.rewards import build_reward
from slm.sft import render_prompt
from slm.utils import pick_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default="out/ckpt.pt", help="model to start from")
    parser.add_argument("--prompts", help="JSON lines of {prompt, answer}: chat-format task prompts")
    parser.add_argument("--data", help="dataset directory: random text snippets as prompts")
    parser.add_argument("--out", default="out/grpo.pt")
    parser.add_argument("--reward", action="append", required=True,
                        help="reward to maximize; repeat to add several (see slm/rl/rewards.py)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8, help="completions sampled per prompt")
    parser.add_argument("--prompt-len", type=int, default=32, help="snippet length, for --data prompts")
    parser.add_argument("--completion-len", type=int, default=64, help="maximum completion length")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--updates-per-step", type=int, default=2, help="gradient steps on each batch")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=0.2, help="PPO clipping range")
    parser.add_argument("--kl-weight", type=float, default=0.05, help="pull toward the reference model")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    assert args.prompts or args.data, "give --prompts (task prompts) or --data (text snippets)"
    return args


@dataclass
class Prompt:
    token_ids: list
    answer: str = None  # known-correct answer, for verifiable rewards


class PromptSource:
    """Hands out batches of prompts, from a task file or from random text snippets."""

    def __init__(self, args, tokenizer, device):
        self.args, self.tokenizer, self.device = args, tokenizer, device
        if args.prompts:
            self.tasks = load_jsonl(args.prompts)
        else:
            self.tokens = load_tokens(args.data, "train", tokenizer.vocab_size)

    def sample(self, count):
        if self.args.prompts:
            chosen = random.sample(self.tasks, count)
            return [Prompt(self.tokenizer.encode(render_prompt([{"role": "user", "content": t["prompt"]}])),
                           t["answer"]) for t in chosen]
        windows = random_windows(self.tokens, count, self.args.prompt_len, self.device)
        return [Prompt(w.tolist()) for w in windows]


@dataclass
class Group:
    """Everything GRPO needs about one prompt and its sampled completions."""
    sequences: torch.Tensor  # (group, prompt_len + completion_len) token ids
    prompt_len: int
    mask: torch.Tensor  # (group, completion_len) True for real completion tokens
    old_logprobs: torch.Tensor
    reference_logprobs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    completions: list  # decoded completion texts


@torch.no_grad()
def sample_group(prompt, policy, reference, tokenizer, reward_fn, args, device):
    prompt_ids = torch.tensor([prompt.token_ids], device=device).repeat(args.group_size, 1)
    prompt_len = prompt_ids.shape[1]
    assert prompt_len + args.completion_len <= policy.config.block_size, "prompt + completion too long"

    sequences = policy.generate(prompt_ids, args.completion_len, args.temperature, eos_id=tokenizer.eos_id)
    completion_ids = sequences[:, prompt_len:]
    mask = completion_mask(completion_ids, tokenizer.eos_id)

    completions = []
    for ids, keep in zip(completion_ids.tolist(), mask.tolist()):
        real = [t for t, k in zip(ids, keep) if k and t != tokenizer.eos_id]
        completions.append(tokenizer.decode(real))
    prompt_text = tokenizer.decode(prompt.token_ids)
    rewards = torch.tensor(
        reward_fn([prompt_text] * args.group_size, completions, [prompt.answer] * args.group_size),
        device=device,
    )
    return Group(
        sequences=sequences, prompt_len=prompt_len, mask=mask,
        old_logprobs=completion_logprobs(policy, sequences, prompt_len),
        reference_logprobs=completion_logprobs(reference, sequences, prompt_len),
        rewards=rewards, advantages=group_advantages(rewards), completions=completions,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = pick_device(args.device)

    # The policy is the model we train. The reference is a frozen copy of where it started.
    policy, tokenizer, _ = load_checkpoint(args.ckpt, device, dropout=0.0)
    reference, _, _ = load_checkpoint(args.ckpt, device, dropout=0.0)
    reference.requires_grad_(False)
    policy.eval()  # dropout is off; eval mode just makes that explicit
    reference.eval()

    prompts = PromptSource(args, tokenizer, device)
    reward_fn = build_reward(args.reward, args.data, tokenizer, device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)

    print(f"device {device} | rewards {args.reward} | "
          f"{args.prompts_per_step} prompts x {args.group_size} completions per step")
    start_time = time.time()

    for step in range(1, args.steps + 1):
        # 1-3: sample and score a group of completions for each prompt.
        groups = [sample_group(p, policy, reference, tokenizer, reward_fn, args, device)
                  for p in prompts.sample(args.prompts_per_step)]

        # 4: push the model toward the better completions in each group.
        for _ in range(args.updates_per_step):
            optimizer.zero_grad(set_to_none=True)
            all_stats = []
            for group in groups:
                logprobs = completion_logprobs(policy, group.sequences, group.prompt_len)
                loss, stats = grpo_loss(logprobs, group.old_logprobs, group.reference_logprobs,
                                        group.advantages, group.mask, args.clip, args.kl_weight)
                (loss / len(groups)).backward()  # average the loss over groups
                all_stats.append(stats)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

        if step % args.log_interval == 0 or step == 1:
            mean_reward = torch.cat([g.rewards for g in groups]).mean().item()
            kl = sum(s["kl"] for s in all_stats) / len(all_stats)
            clipped = sum(s["clipped_fraction"] for s in all_stats) / len(all_stats)
            print(f"step {step:4d} | reward {mean_reward:.3f} | kl {kl:.4f} "
                  f"| clipped {clipped:.2f} | {time.time() - start_time:.0f}s")
            print(f"    sample: {groups[0].completions[0]!r}")

    save_checkpoint(args.out, policy, tokenizer, rl_method="grpo", rewards=args.reward)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
