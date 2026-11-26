"""Supervised fine-tuning (SFT): teach a pretrained model to answer prompts.

Pretraining teaches the model to continue text. SFT shows it examples of
conversations, so it learns to reply to a user and then stop.

Data format: JSON lines, either a single exchange

    {"prompt": "Uppercase: hello", "response": "HELLO"}

or a full conversation

    {"messages": [{"role": "system",    "content": "..."},
                  {"role": "user",      "content": "..."},
                  {"role": "assistant", "content": "..."}, ...]}

Chat template: each conversation is turned into plain text like

    User: Uppercase: hello
    Assistant: HELLO<|endoftext|>

Loss masking: the model is only trained to predict the assistant's replies
(and the EOS after each one). It still reads the user's text, but isn't
graded on predicting it; that's not its job.
"""

import random

import torch

from slm.tokenizer import EOS

ROLE_PREFIXES = {"system": "System: ", "user": "User: ", "assistant": "Assistant: "}
IGNORE = -100  # target value that cross_entropy skips


def to_messages(example):
    """Accept either format and return a list of {"role", "content"} messages."""
    if "messages" in example:
        return example["messages"]
    response = example.get("response", example.get("answer"))
    return [{"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": response}]


def render_prompt(messages):
    """Text to feed the model so that it writes the next assistant reply."""
    text = ""
    for message in messages:
        # Earlier assistant replies end with EOS, exactly as in training.
        end = EOS + "\n" if message["role"] == "assistant" else "\n"
        text += ROLE_PREFIXES[message["role"]] + message["content"] + end
    return text + ROLE_PREFIXES["assistant"]


def tokenize_conversation(messages, tokenizer):
    """Token ids for a whole conversation, plus which ones the model is trained on.

    Returns (ids, trainable) where trainable[i] says whether ids[i] is part of
    an assistant reply.
    """
    ids, trainable = [], []

    def add(text, train):
        new = tokenizer.encode(text)
        ids.extend(new)
        trainable.extend([train] * len(new))

    for message in messages:
        if message["role"] == "assistant":
            add(ROLE_PREFIXES["assistant"], False)
            add(message["content"], True)
            ids.append(tokenizer.eos_id)  # learning to stop is part of the reply
            trainable.append(True)
            add("\n", False)
        else:
            add(ROLE_PREFIXES[message["role"]] + message["content"] + "\n", False)
    return ids, trainable


class SFTDataset:
    """Tokenized conversations, cut to fit the model's context window."""

    def __init__(self, examples, tokenizer, block_size):
        self.items = []
        for example in examples:
            ids, trainable = tokenize_conversation(to_messages(example), tokenizer)
            ids, trainable = ids[: block_size + 1], trainable[: block_size + 1]
            if any(trainable[1:]):  # skip examples whose reply was cut off entirely
                self.items.append((ids, trainable))

    def __len__(self):
        return len(self.items)

    def get_batch(self, batch_size, device):
        """A random padded batch. Targets are IGNORE except on assistant tokens."""
        chosen = random.choices(self.items, k=batch_size)
        length = max(len(ids) for ids, _ in chosen) - 1
        inputs = torch.zeros((batch_size, length), dtype=torch.long)
        targets = torch.full((batch_size, length), IGNORE, dtype=torch.long)
        for row, (ids, trainable) in enumerate(chosen):
            ids_tensor = torch.tensor(ids)
            n = len(ids) - 1
            inputs[row, :n] = ids_tensor[:-1]
            row_targets = ids_tensor[1:].clone()
            row_targets[~torch.tensor(trainable[1:])] = IGNORE
            targets[row, :n] = row_targets
        return inputs.to(device), targets.to(device)


@torch.no_grad()
def generate_reply(model, tokenizer, messages, max_new_tokens=100, temperature=0.0, top_k=None):
    """Generate the next assistant reply to a conversation, as text."""
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(render_prompt(messages))
    prompt_ids = prompt_ids[-(model.config.block_size - 1):]  # keep the most recent context
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = model.generate(prompt, max_new_tokens, temperature, top_k, eos_id=tokenizer.eos_id)
    reply_ids = output[0, len(prompt_ids):].tolist()
    if tokenizer.eos_id in reply_ids:
        reply_ids = reply_ids[: reply_ids.index(tokenizer.eos_id)]
    return tokenizer.decode(reply_ids)
