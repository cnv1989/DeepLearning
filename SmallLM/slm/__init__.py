"""slm: a small language model, built from scratch.

Modules
-------
model        The transformer (RMSNorm, RoPE, SwiGLU, grouped-query attention, KV cache).
tokenizer    Character and GPT-2 tokenizers, and the shared end-of-text token.
bpe          Training your own byte-level BPE tokenizer.
data         Reading tokenized datasets and JSON-lines files; sampling batches.
trainer      The training loop shared by pretraining and fine-tuning.
config       Settings from defaults, JSON config files and command-line flags.
distributed  Multi-GPU training setup.
metrics      Logging training metrics.
sft          Chat template and loss masking for supervised fine-tuning.
tasks        Synthetic tasks with checkable answers.
checkpoint   Saving and loading models together with their tokenizer.
utils        Device, precision and learning-rate schedule.
rl           Reinforcement learning: rewards, reward model, GRPO and DPO.
"""
