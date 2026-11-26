# Lesson 4: The model, a tiny character-level language model

## What it does

Given the last **N** characters, predict the next one. Generate text by
feeding each prediction back in. That's the whole "language model" loop.
GPT works the same way, just with a far bigger network and tokens instead of
characters.

## Architecture (`tt` config)

```
context: 3 tokens (5 bits each, 32-symbol alphabet " abc...xyz.,'!?")
   │
   ▼  embedding lookup: each token → 4 signed 4-bit numbers
x: 12 integers
   │
   ▼  layer 1: 16 neurons, ternary weights W1[16][12], bias b1
   ▼  ReLU, then h = round(acc / 4), clamped to 0..15
h: 16 small integers
   │
   ▼  layer 2: 32 outputs, ternary weights W2[32][16], bias b2
z: 32 integer logits, one per character
   │
   ▼  (+ optional random noise) → argmax
next token
```

This is the "neural probabilistic language model" (Bengio et al., 2003):
embeddings plus an MLP over a fixed window. No attention, so the memory is just
the window. Lesson 10 discusses what attention would cost in hardware.

Size: 128 embedding values + 192 + 512 ternary weights + 48 biases, about
**1.8 kbit** of weights. A Taalas chip holds billions of parameters; this holds
about 900. Same idea, scaled down 10 million times.

## Quantisation-aware training (QAT)

If you train in float and round at the end, a model this small falls apart.
Instead, `model/train.py` runs the exact integer hardware arithmetic **in the
forward pass** during training:

```python
x  = round(E)[context]               # integer embeddings, clipped to ±7
a1 = x @ ternary(W1).T + round(b1)   # integer
h  = clip(round(relu(a1) / 4), 0, 15)
z  = h @ ternary(W2).T + round(b2)   # integer logits
loss = cross_entropy(tau * z, target)   # tau: a learned temperature, training only
```

`round` and `ternary` have zero gradient almost everywhere, so how does
anything learn? With the **straight-through estimator (STE)**: in the backward
pass we pretend they're the identity function and pass gradients straight
through to the hidden float "shadow" weights. The float weights drift
slowly, and every so often one of them crosses a rounding threshold and the
integer weight the hardware sees changes.

Ternarisation follows Ternary Weight Networks: values with
`|w| < 0.7 × mean|w|` become 0, the rest become ±1.

The backward pass is written out by hand in `QATModel.backward` (about 25 lines
of NumPy). If backprop is rusty, compare each line with the forward pass;
every `*mask` is an STE or a ReLU/clip derivative.

## Run it

```bash
python3 model/train.py --config tt
```

```
step     0  loss 3.478  next-char accuracy   3.8%
...
step  4000  loss 2.091  next-char accuracy  39.4%
params: E 32x4, W1 16x12 (92 non-zero), W2 32x16 (265 non-zero)
sample (noise=off): the 'came came came came ...'
sample (noise=on ): the 'to and the the and and hee. the cau t the cae and ...'
```

A few things to notice:
- **Greedy decoding loops** ("came came came"). That's why the hardware has a
  sampling mode (lesson 7).
- About **half the weights are zero**. Zeros cost nothing in the hard-wired
  version (lesson 5), so sparsity is free area.
- Around 38% accuracy from a 3-character window is reasonable. The `fpga`
  config (6 characters, 48 neurons) reaches about 53%.

At the end, training checks that the integer reference model
(`model/ref_model.py`) gives *identical* logits to the training forward pass.
That check is what lets you trust everything downstream.

## Using your own text

```bash
python3 model/train.py --config tt --corpus my_text.txt
make gen test
```

Anything works (song lyrics, code comments, a book). Upper case folds to
lower case, and anything outside the 32-symbol alphabet becomes a space.
A few kilobytes is enough; a model this small can't use much more.

## Exercises

1. Train with `--seed 2`, `3`, `4`. How much does accuracy vary? That spread
   is your noise floor when comparing changes.
2. Add a config in `model/config.py` with `H=8`. How much accuracy do you
   lose? With `N=4`?
3. Replace `ternary()` with a 2-bit quantiser ({−2, −1, 1, 2}). What would the
   hardware MAC need? (Hint: a shift.)

Next: [Lesson 5: Hard-wiring the weights](05_hardwiring_weights.md)
