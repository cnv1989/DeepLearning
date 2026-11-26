# 5. Pretraining

**Code:** [`train.py`](../train.py), [`slm/trainer.py`](../slm/trainer.py), [`slm/utils.py`](../slm/utils.py), [`slm/config.py`](../slm/config.py), [`slm/metrics.py`](../slm/metrics.py), [`configs/`](../configs)

## What it is

Pretraining teaches the model language by giving it one simple task, billions of
times: **predict the next token.** Nothing else. Grammar, facts, style and some
reasoning all emerge because they help with that prediction.

```bash
python train.py --config configs/tiny.json --data data/shakespeare_char
```

## The objective: cross-entropy loss

At each position, the model gives the correct next token some probability *p*.
The loss is **−log p**, averaged over all positions in the batch:

| Probability given to the right token | Loss |
|---|---|
| 1.0 (certain and right) | 0 |
| 0.5 | 0.69 |
| 1/66 (random guess, 66 characters) | 4.19 |
| 0.001 (confident and wrong) | 6.9 |

Confidently wrong answers are punished hardest.

**Perplexity** = e^loss is easier to picture: the model is as unsure as if it
were choosing uniformly among *that many* tokens. The tiny model went from
perplexity 66 (random) to 4.9 in 6 minutes. At each character, it's about as
unsure as choosing among 5 options.

## One training step (`Trainer.train_step`)

```
1. take a batch            inputs, targets = get_batch("train")
2. forward pass            logits, loss = model(inputs, targets)
3. backward pass           loss.backward()          → a gradient for every weight
4. clip gradients          scale them down if their total size exceeds grad_clip
5. optimizer step          nudge every weight against its gradient
6. zero the gradients      ready for the next step
```

### The optimizer: AdamW (`configure_optimizer`)

Plain gradient descent moves every weight by `lr × gradient`. **Adam** also
tracks a running average of each weight's gradient (momentum: keep going in a
consistent direction) and of its square (scale: take smaller steps for weights
with noisy gradients). Every weight effectively gets its own step size.

The **W** is weight decay: each step, weights shrink slightly toward zero
(×(1 − lr·0.1)), which discourages over-large weights and overfitting. It's
applied to weight matrices only, not to norm gains, where shrinking would
just fight the normalization.

### The learning-rate schedule (`cosine_lr`)

```
 lr
 max ┤   ╭──╮
     │  ╱    ╲_
     │ ╱       ╲__
 min ┤╱           ╲____
     └─┬───────────────── step
     warmup
```

* **Warmup** (`warmup_iters`): start tiny and ramp up. At the start, Adam's
  running averages are unreliable and large steps can wreck the model.
* **Cosine decay** to `min_lr`: big steps early to make fast progress, small
  steps late to settle into a good spot.

The learning rate is the single most important setting. Too high: the loss
spikes or becomes NaN. Too low: training crawls.

### Gradient clipping (`grad_clip`)

Occasionally one batch produces a huge gradient, and one huge step can undo
hours of progress. Clipping rescales the whole gradient so its total size
(norm) is at most 1.0.

### Gradient accumulation (`grad_accum`)

Bigger batches give smoother gradients but need more memory. With
`--grad-accum 8`, the trainer runs 8 small batches, adds up their gradients,
and takes one step: the same as one batch 8× bigger, at the memory cost of one.
Each loss is divided by 8 so the sum is an average.

## Mixed precision (`pick_dtype`, `autocast`)

Weights are stored in 32-bit floats, but most of the math can run in 16-bit,
which is about 2× faster on modern GPUs and uses half the activation memory.

| `--dtype` | Used on | Notes |
|---|---|---|
| `bfloat16` | recent NVIDIA GPUs (A100, H100, RTX 30/40) | same range as float32, less precision; just works |
| `float16` | older GPUs | small range, so tiny gradients round to zero; needs **loss scaling** |
| `float32` | CPU, Apple Silicon | full precision |

`auto` picks for you. **Loss scaling** (`GradScaler`, float16 only) multiplies
the loss by a large number before backward so gradients stay representable,
then divides them back before the optimizer step. It automatically skips steps
that overflow.

## Evaluation and checkpoints (`evaluate_and_save`)

Every `eval_interval` steps the trainer measures the average loss on
`eval_iters` random batches from *both* the training and validation sets:

```
step  1000 | train loss 1.4251 | val loss 1.6354 | val perplexity 5.13 | 259s
```

* Both falling: learning.
* Train falling, val flat or rising: **overfitting** (memorizing the training text).
* Both flat: learning rate too low, model too small, or finished.

Two checkpoints are written:

* `out/ckpt.pt`: the **best** model so far (lowest validation loss). Use this one.
* `out/last.pt`: the **latest** state, including the optimizer. `--resume`
  continues from here after a crash or to train longer.

Each checkpoint holds the model config, weights, tokenizer and training
settings, so it's fully self-contained.

## Settings and configs (`slm/config.py`)

Settings come from three layers, later ones winning:

```
defaults in train.py   <   configs/small.json   <   command-line flags
```

```bash
python train.py --config configs/small.json --data data/tinystories_bpe --lr 6e-4 --max-iters 10000
```

A config file with an unknown setting is an error, so typos don't silently
do nothing.

## Metrics (`slm/metrics.py`)

Every `log_interval` steps, a line is appended to `out/metrics.jsonl`: training
loss, learning rate and throughput (tokens per second). Validation losses are
added at each evaluation. Plot them with:

```python
import pandas as pd
df = pd.read_json("out/metrics.jsonl", lines=True)
df.plot(x="step", y="train_loss")
```

Add `--wandb` (after `pip install wandb`) to see live charts in Weights &
Biases.

## Speed

* `--compile` runs `torch.compile`, fusing operations into faster GPU kernels
  (first steps are slow while it compiles).
* On CUDA, the optimizer uses the fused AdamW implementation.
* Throughput (`tokens_per_sec` in the metrics) is the number to watch when
  tuning batch size and settings.

## Common problems

| Symptom | Likely cause | Try |
|---|---|---|
| loss is NaN or explodes | learning rate too high | lower `lr`, more `warmup_iters` |
| loss stuck near ln(vocab) | learning rate far too low, or a bug | raise `lr`; check data looks right |
| val loss rises while train falls | overfitting | more data, more `dropout`, smaller model |
| out of memory | batch or model too big | lower `batch_size` and raise `grad_accum`; `--activation-checkpointing` |
| very slow on GPU | float32 or no compile | `--dtype auto`, `--compile` |

## Try it

```bash
python train.py --config configs/tiny.json --data data/shakespeare_char --lr 0.01
python train.py --config configs/tiny.json --data data/shakespeare_char --lr 0.0001
```

Watch how the loss behaves when the learning rate is 10× too high, then 10× too low.
