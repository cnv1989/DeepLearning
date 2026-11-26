# SmallLM guides

One guide per component, in the order a model is built. Each explains what the
component is, why it matters, how it works (with worked examples), where it
lives in the code, which settings to change, common problems, and something
to try.

| # | Guide | You'll learn |
|---|---|---|
| 1 | [Tokenization](01-tokenization.md) | how text becomes numbers; how BPE learns a vocabulary; the end-of-text token |
| 2 | [The data pipeline](02-data-pipeline.md) | reading, cleaning and deduplicating text; how batches are sampled; how much data you need |
| 3 | [The model](03-model-architecture.md) | attention, residual blocks, RMSNorm, RoPE, SwiGLU, grouped-query attention, initialization |
| 4 | [Generation](04-generation.md) | temperature and top-k; the KV cache; stopping at EOS |
| 5 | [Pretraining](05-pretraining.md) | the loss, AdamW, the learning-rate schedule, mixed precision, checkpoints, reading the logs |
| 6 | [Multi-GPU training](06-distributed-training.md) | data parallelism with DDP; effective batch size; what comes after DDP |
| 7 | [Supervised fine-tuning](07-supervised-fine-tuning.md) | chat templates; training only on answers; synthetic tasks |
| 8 | [Evaluation](08-evaluation.md) | perplexity and task accuracy; what makes an evaluation trustworthy |
| 9 | [Reinforcement learning with GRPO](09-reinforcement-learning-grpo.md) | rewards, group advantages, clipping and the KL penalty; reward hacking; what the test runs taught |
| 10 | [Reward models and DPO](10-reward-models-and-dpo.md) | learning from preferences; Bradley-Terry; DPO vs reward-model RL |

**Where to start:** if transformers are new to you, read 3, then 1, 2 and 5.
If you know transformers and want the post-training side, read 7, 9 and 10.

The numbers quoted in the guides come from real runs of this code on a 4-core
CPU with the `tiny` config (see the [main README](../README.md#results-from-a-test-run-4-core-cpu)).
