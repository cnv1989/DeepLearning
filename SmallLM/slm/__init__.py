"""slm: a small language model, built from scratch.

Modules
-------
model       The GPT transformer.
tokenizer   Turning text into token ids and back.
data        Reading tokenized datasets and sampling training batches.
checkpoint  Saving and loading models together with their tokenizer.
utils       Device selection and the learning-rate schedule.
rl          Reinforcement learning: rewards, reward model, GRPO and DPO.
"""
