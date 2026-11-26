"""Log-probabilities of token sequences under a model.

Every RL method needs the same quantity: how likely the model is to produce
the tokens it (or someone) produced. For a sequence t_0 ... t_n:

    log p(t_i | t_0 ... t_{i-1})    for each i >= 1
"""

import torch
import torch.nn.functional as F


def token_logprobs(model, tokens):
    """Log-probability of each token given the tokens before it.

    tokens:  (batch, seq_len)
    returns: (batch, seq_len - 1); entry [b, i] is log p(tokens[b, i+1] | tokens[b, :i+1])
    """
    logits, _ = model(tokens[:, :-1])
    log_probs = F.log_softmax(logits.float(), dim=-1)
    next_tokens = tokens[:, 1:].unsqueeze(-1)
    return log_probs.gather(dim=-1, index=next_tokens).squeeze(-1)


def completion_logprobs(model, tokens, prompt_len):
    """Per-token log-probabilities of just the completion (the part after the prompt).

    All sequences share the same prompt length and completion length.
    returns: (batch, completion_len)
    """
    return token_logprobs(model, tokens)[:, prompt_len - 1:]


def completion_mask(completions, eos_id):
    """Which completion tokens count: everything up to and including the first EOS.

    `generate` pads finished sequences with EOS; those padding tokens weren't
    really chosen by the model, so they're excluded from the loss.
    completions: (batch, completion_len) -> bool mask of the same shape
    """
    is_eos = completions == eos_id
    eos_seen_before = (is_eos.cumsum(dim=1) - is_eos.long()) > 0
    return ~eos_seen_before


def truncate_pairs(prompts, completions, max_len):
    """Make every prompt + completion fit in `max_len` tokens.

    Completions are cut at the end; prompts are cut at the start, so the
    text right before the completion is kept.
    """
    new_prompts, new_completions = [], []
    for prompt, completion in zip(prompts, completions):
        completion = completion[: max_len - 1]
        room_for_prompt = max_len - len(completion)
        new_prompts.append(prompt[-room_for_prompt:])
        new_completions.append(completion)
    return new_prompts, new_completions


def pad_prompt_completion_pairs(prompts, completions, device, pad_id=0):
    """Pack variable-length (prompt, completion) pairs into one padded batch.

    prompts, completions: lists of token-id lists
    returns:
        tokens           (batch, max_len) prompt + completion, right-padded
        completion_mask  (batch, max_len) True where the token belongs to the completion
        lengths          (batch,) real length of each sequence before padding
    """
    sequences = [p + c for p, c in zip(prompts, completions)]
    max_len = max(len(s) for s in sequences)
    batch = len(sequences)

    tokens = torch.full((batch, max_len), pad_id, dtype=torch.long)
    completion_mask = torch.zeros((batch, max_len), dtype=torch.bool)
    for i, (prompt, seq) in enumerate(zip(prompts, sequences)):
        tokens[i, : len(seq)] = torch.tensor(seq)
        completion_mask[i, len(prompt) : len(seq)] = True

    lengths = torch.tensor([len(s) for s in sequences])
    return tokens.to(device), completion_mask.to(device), lengths.to(device)


def completion_logprob_sums(model, tokens, completion_mask):
    """Total log-probability of each completion: log p(completion | prompt).

    Padding and prompt tokens are excluded by `completion_mask`.
    returns: (batch,)
    """
    per_token = token_logprobs(model, tokens)
    # token_logprobs drops the first position, so drop it from the mask too.
    mask = completion_mask[:, 1:].float()
    return (per_token * mask).sum(dim=1)
