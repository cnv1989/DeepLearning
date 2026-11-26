"""Synthetic tasks with checkable answers.

Real instruction data is large and messy. These tiny tasks have exactly one
correct answer, so they are ideal for seeing each training stage work:

    SFT   learns the format and gets some answers right
    GRPO  with an exact_match reward then improves accuracy further
    evaluate.py measures accuracy

Each example looks like  {"prompt": "Reverse: rose", "answer": "esor"}.
"""

import random

TASKS = {
    "uppercase": lambda word: word.upper(),
    "reverse": lambda word: word[::-1],
    "repeat": lambda word: f"{word} {word}",
    "first letter": lambda word: word[0],
}


def make_example(task, word):
    return {"prompt": f"{task.capitalize()}: {word}", "answer": TASKS[task](word), "task": task}


def make_examples(words, count, tasks=None, seed=0):
    rng = random.Random(seed)
    tasks = tasks or list(TASKS)
    return [make_example(rng.choice(tasks), rng.choice(words)) for _ in range(count)]
