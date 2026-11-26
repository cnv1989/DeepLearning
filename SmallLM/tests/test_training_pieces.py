import math

import torch

from slm.rl.dpo import dpo_loss
from slm.rl.grpo import group_advantages, grpo_loss, kl_to_reference
from slm.rl.logprobs import completion_mask, pad_prompt_completion_pairs, truncate_pairs
from slm.rl.rewards import build_reward
from slm.sft import IGNORE, SFTDataset, render_prompt, tokenize_conversation
from slm.tokenizer import EOS, CharTokenizer
from slm.utils import cosine_lr

TOKENIZER = CharTokenizer.from_text("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ :\n")


# ---------------------------------------------------------------- SFT


def test_only_assistant_tokens_are_trained():
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    ids, trainable = tokenize_conversation(messages, TOKENIZER)
    trained_text = TOKENIZER.decode([i for i, t in zip(ids, trainable) if t])
    assert trained_text == "yo" + EOS


def test_prompt_matches_training_format():
    messages = [{"role": "user", "content": "hi"}]
    ids, _ = tokenize_conversation(messages + [{"role": "assistant", "content": "yo"}], TOKENIZER)
    prompt_ids = TOKENIZER.encode(render_prompt(messages))
    assert ids[: len(prompt_ids)] == prompt_ids


def test_sft_batch_targets():
    dataset = SFTDataset([{"prompt": "hi", "response": "yo"}], TOKENIZER, block_size=64)
    inputs, targets = dataset.get_batch(2, "cpu")
    real_targets = targets[0][targets[0] != IGNORE].tolist()
    assert TOKENIZER.decode(real_targets) == "yo" + EOS


# ---------------------------------------------------------------- RL


def test_group_advantages():
    advantages = group_advantages(torch.tensor([1.0, 2.0, 3.0]))
    assert abs(advantages.mean().item()) < 1e-6
    assert advantages[2] > advantages[1] > advantages[0]
    assert torch.allclose(group_advantages(torch.ones(4)), torch.zeros(4))


def test_kl_is_zero_for_identical_models():
    logprobs = torch.randn(3, 5)
    assert torch.allclose(kl_to_reference(logprobs, logprobs), torch.zeros(3, 5))


def test_grpo_gradient_favors_positive_advantage():
    logprobs = torch.zeros(2, 4, requires_grad=True)
    mask = torch.ones(2, 4, dtype=torch.bool)
    loss, _ = grpo_loss(logprobs, logprobs.detach(), logprobs.detach(), torch.tensor([1.0, -1.0]), mask)
    loss.backward()
    assert (logprobs.grad[0] < 0).all()  # descending the loss raises completion 0's probability
    assert (logprobs.grad[1] > 0).all()


def test_completion_mask_stops_after_first_eos():
    completions = torch.tensor([[5, 9, 7, 9], [1, 2, 3, 4]])
    mask = completion_mask(completions, eos_id=9)
    assert mask.tolist() == [[True, True, False, False], [True, True, True, True]]


def test_dpo_loss_starts_at_log_two():
    zeros = torch.zeros(4)
    loss, stats = dpo_loss(zeros, zeros, zeros, zeros)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)


def test_padding_and_truncation():
    prompts, completions = truncate_pairs([[1, 2, 3, 4]], [[5, 6]], max_len=4)
    assert prompts == [[3, 4]] and completions == [[5, 6]]
    tokens, mask, lengths = pad_prompt_completion_pairs([[1], [1, 2]], [[3], [4, 5]], "cpu")
    assert tokens.tolist() == [[1, 3, 0, 0], [1, 2, 4, 5]]
    assert mask.tolist() == [[False, True, False, False], [False, False, True, True]]
    assert lengths.tolist() == [2, 4]


def test_verifiable_rewards():
    reward = build_reward(["exact_match", "similarity"], None, TOKENIZER, "cpu")
    right, wrong = reward(["p", "p"], ["ROSE", "XYZ"], ["ROSE", "ROSE"])
    assert right == 2.0
    assert wrong < 1.0


# ---------------------------------------------------------------- schedule


def test_cosine_schedule():
    assert cosine_lr(0, 1.0, 0.1, warmup_steps=10, total_steps=100) == 0.1
    assert cosine_lr(9, 1.0, 0.1, warmup_steps=10, total_steps=100) == 1.0
    assert math.isclose(cosine_lr(100, 1.0, 0.1, warmup_steps=10, total_steps=100), 0.1)
