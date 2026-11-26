"""Generate text from any checkpoint (pretrained, SFT, GRPO or DPO).

    python sample.py --prompt "ROMEO:" --tokens 500          continue text
    python sample.py --ckpt out/sft/ckpt.pt --chat --prompt "Reverse: rose"
    python sample.py --ckpt out/sft/ckpt.pt --chat           interactive chat

Generation uses a KV cache, so each new token costs one position.
"""

import argparse

import torch

from slm.checkpoint import load_checkpoint
from slm.sft import generate_reply
from slm.utils import pick_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default="out/ckpt.pt")
    parser.add_argument("--prompt", help="text to continue (or the user message, with --chat)")
    parser.add_argument("--chat", action="store_true", help="use the chat format from fine-tuning")
    parser.add_argument("--tokens", type=int, default=500, help="maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="0 = always the most likely token")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def continue_text(model, tokenizer, args, device):
    prompt_ids = tokenizer.encode(args.prompt or "\n") or [tokenizer.eos_id]
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for i in range(args.num_samples):
        output = model.generate(prompt, args.tokens, args.temperature, args.top_k)
        print(tokenizer.decode(output[0].tolist()))
        if i < args.num_samples - 1:
            print("-" * 60)


def chat(model, tokenizer, args):
    def reply(messages):
        return generate_reply(model, tokenizer, messages, args.tokens, args.temperature, args.top_k)

    if args.prompt:
        print(reply([{"role": "user", "content": args.prompt}]))
        return

    print("Chatting. Empty line to quit.")
    messages = []
    while True:
        user_text = input("you> ").strip()
        if not user_text:
            break
        messages.append({"role": "user", "content": user_text})
        answer = reply(messages)
        messages.append({"role": "assistant", "content": answer})
        print(f"model> {answer}")


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = pick_device(args.device)
    model, tokenizer, _ = load_checkpoint(args.ckpt, device, dropout=0.0)
    model.eval()

    if args.chat:
        chat(model, tokenizer, args)
    else:
        continue_text(model, tokenizer, args, device)


if __name__ == "__main__":
    main()
