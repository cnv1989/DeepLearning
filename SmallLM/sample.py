"""Generate text from any checkpoint (pretrained, GRPO or DPO).

    python sample.py --prompt "ROMEO:" --tokens 500
    python sample.py --ckpt out/grpo.pt --prompt "ROMEO:"
"""

import argparse

import torch

from slm.checkpoint import load_checkpoint
from slm.utils import pick_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default="out/ckpt.pt")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--tokens", type=int, default=500, help="how many tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = pick_device(args.device)

    model, tokenizer, _ = load_checkpoint(args.ckpt, device, dropout=0.0)
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt) or [0]
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for i in range(args.num_samples):
        output = model.generate(prompt, args.tokens, temperature=args.temperature, top_k=args.top_k)
        print(tokenizer.decode(output[0].tolist()))
        if i < args.num_samples - 1:
            print("-" * 60)


if __name__ == "__main__":
    main()
