"""Reward functions: score generated text with a number. Higher is better.

Every reward function takes a list of prompts and a list of completions
(both plain strings) and returns one float per completion.

Built-in rewards (choose them with --reward on the command line):

    known_words        fraction of words that appear in the training text
    distinct_words     fraction of words that are not repeats
    keyword=WORD       1.0 if the completion contains WORD, else 0.0
    model=PATH         score from a trained reward model (see reward_model.py)

Several rewards can be combined; their scores are added together.

A note on reward hacking: RL maximizes exactly what you measure. With only
`known_words`, the model learns to repeat "the the the ...", because every
word is real. Adding `distinct_words` closes that loophole. Always read
samples, not just reward curves.
"""

import re

import torch

from slm.data import load_tokens

WORD_PATTERN = re.compile(r"[a-z']+")


def words_in(text):
    return WORD_PATTERN.findall(text.lower())


class KnownWordsReward:
    """Fraction of words in the completion that appear in a reference vocabulary."""

    def __init__(self, vocabulary):
        self.vocabulary = set(vocabulary)

    @classmethod
    def from_dataset(cls, data_dir, tokenizer):
        # Rebuild the training text from its tokens and collect every word in it.
        tokens = load_tokens(data_dir, "train", tokenizer.vocab_size)
        text = tokenizer.decode(tokens.tolist())
        return cls(words_in(text))

    def __call__(self, prompts, completions):
        scores = []
        for text in completions:
            words = words_in(text)
            known = sum(w in self.vocabulary for w in words)
            scores.append(known / len(words) if words else 0.0)
        return scores


class DistinctWordsReward:
    """Fraction of words in the completion that are unique. Penalizes repetition."""

    def __call__(self, prompts, completions):
        scores = []
        for text in completions:
            words = words_in(text)
            scores.append(len(set(words)) / len(words) if words else 0.0)
        return scores


class KeywordReward:
    """1.0 if the completion mentions the keyword, otherwise 0.0."""

    def __init__(self, keyword):
        self.keyword = keyword.lower()

    def __call__(self, prompts, completions):
        return [1.0 if self.keyword in words_in(text) else 0.0 for text in completions]


class LearnedReward:
    """Score text with a trained RewardModel."""

    def __init__(self, path, device):
        from slm.rl.reward_model import load_reward_model

        self.model, self.tokenizer = load_reward_model(path, device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, prompts, completions):
        scores = self.model.score_texts(self.tokenizer, prompts, completions, self.device)
        return scores.tolist()


class SumOfRewards:
    """Adds up several reward functions."""

    def __init__(self, rewards):
        self.rewards = rewards

    def __call__(self, prompts, completions):
        total = [0.0] * len(completions)
        for reward in self.rewards:
            for i, score in enumerate(reward(prompts, completions)):
                total[i] += score
        return total


def build_reward(specs, data_dir, tokenizer, device):
    """Turn command-line specs like ["known_words", "keyword=love"] into one reward function."""
    rewards = []
    for spec in specs:
        name, _, argument = spec.partition("=")
        if name == "known_words":
            rewards.append(KnownWordsReward.from_dataset(data_dir, tokenizer))
        elif name == "distinct_words":
            rewards.append(DistinctWordsReward())
        elif name == "keyword":
            rewards.append(KeywordReward(argument))
        elif name == "model":
            rewards.append(LearnedReward(argument, device))
        else:
            raise ValueError(f"unknown reward '{spec}'")
    return SumOfRewards(rewards)
