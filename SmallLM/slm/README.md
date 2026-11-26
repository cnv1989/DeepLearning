# `slm/`: the library

The reusable parts of SmallLM. The scripts in the folder above are thin
wrappers that read settings and call into here. Each file links to the
guide that explains it.

| File | What it does | Guide |
|---|---|---|
| [`tokenizer.py`](tokenizer.py) | Character and GPT-2 tokenizers; the end-of-text token | [Tokenization](../docs/01-tokenization.md) |
| [`bpe.py`](bpe.py) | Training your own byte-level BPE tokenizer | [Tokenization](../docs/01-tokenization.md) |
| [`data.py`](data.py) | Memory-mapped token files, random batches, JSON-lines files | [Data pipeline](../docs/02-data-pipeline.md) |
| [`model.py`](model.py) | The transformer, KV cache and text generation | [Model](../docs/03-model-architecture.md), [Generation](../docs/04-generation.md) |
| [`trainer.py`](trainer.py) | The training loop shared by pretraining and SFT | [Pretraining](../docs/05-pretraining.md) |
| [`config.py`](config.py) | Settings from defaults, JSON configs and flags | [Pretraining](../docs/05-pretraining.md) |
| [`utils.py`](utils.py) | Device and precision choice, learning-rate schedule | [Pretraining](../docs/05-pretraining.md) |
| [`metrics.py`](metrics.py) | Logging metrics to `metrics.jsonl` and Weights & Biases | [Pretraining](../docs/05-pretraining.md) |
| [`checkpoint.py`](checkpoint.py) | Saving and loading a model together with its tokenizer | [Pretraining](../docs/05-pretraining.md) |
| [`distributed.py`](distributed.py) | Multi-GPU setup with DDP | [Multi-GPU training](../docs/06-distributed-training.md) |
| [`sft.py`](sft.py) | Chat template, loss masking, SFT batches, reply generation | [Supervised fine-tuning](../docs/07-supervised-fine-tuning.md) |
| [`tasks.py`](tasks.py) | Synthetic tasks with checkable answers | [Supervised fine-tuning](../docs/07-supervised-fine-tuning.md) |
| [`rl/`](rl/) | Rewards, GRPO, reward model, DPO | [GRPO](../docs/09-reinforcement-learning-grpo.md), [Preferences](../docs/10-reward-models-and-dpo.md) |
