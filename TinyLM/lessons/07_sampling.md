# Lesson 7: Argmax and sampling

## Greedy decoding gets stuck

Always picking the most likely next character (argmax) is deterministic, and
small models fall into loops: `the came came came came ...`. Language models
avoid this by **sampling**: choosing randomly according to the predicted
probabilities.

## Sampling without a softmax

Real softmax sampling needs `exp()`, a division and a random number compared
against a running sum. That's far too much hardware. There's a neat
alternative:

> **Gumbel-max trick:** `argmax(z + Gumbel noise)` gives a sample from
> `softmax(z)`, exactly.

TinyLM uses a cheap approximation: add **uniform** random noise (0…7 for
`NOISE_BITS = 3`) to each logit, then take the argmax. Logits that are close
to the best get a real chance to win; far-away logits never do. The noise
amplitude works like a **temperature**:
- `NOISE_BITS = 0`: greedy;
- small: mostly sensible, occasionally creative;
- large: gibberish.

Lesson 4's training prints both modes. To try other temperatures without
retraining, override the value stored in the weights file:

```python
import json, sys; sys.path.insert(0, "model")
from ref_model import TinyLM; import charset
w = json.load(open("model/weights/tt.json"))
for nb in (1, 2, 3, 4, 5):
    w["config"]["NOISE_BITS"] = nb
    print(nb, charset.decode(TinyLM(w).generate(charset.encode("the "), 80, noise_en=True)))
```

To change it in hardware, set `NOISE_BITS` in `model/config.py` and run
`make train gen test`.

## Random numbers in hardware: the LFSR

A **linear-feedback shift register** is a shift register whose input is an
XOR of some of its bits. With the right "taps", a 16-bit LFSR cycles through
all 65,535 non-zero states in a pseudo-random order. It costs 16 flip-flops
and one XOR per tap.

```verilog
// Galois form, taps 0xB400 (x^16 + x^14 + x^13 + x^11 + 1)
lfsr_next = {1'b0, lfsr[15:1]} ^ (lfsr[0] ? 16'hB400 : 16'h0000);
```

```python
def lfsr_step(s): return (s >> 1) ^ (0xB400 if s & 1 else 0)
```

## Bit-exact randomness

Randomness and exact verification sound incompatible. They aren't, as long as
the Python model knows exactly **when** the LFSR advances:
- seed `0xACE1` at reset;
- one step per logit (32 per character), whether or not noise is enabled;
- the noise for logit `k` is the low `NOISE_BITS` of the LFSR *before* that
  step.

Since both sides follow those rules, the testbench checks sampled text bit for
bit too (`noise=on` in `tb/vectors`).

A design choice to discuss: on a real chip you might want the LFSR to run
**every clock** so that the user's timing adds real entropy. That gives better
randomness but breaks reproducibility. A common compromise is a "seed" input
or a test-mode pin.

## Argmax

Running argmax over logits that arrive one per 16 cycles:

```verilog
wire take = (k == 0) || (logit > best_val);     // strict > : first maximum wins
```

The tie rule must match Python (`np.argmax` and `ref_model` both keep the
first maximum). Lesson 2's exercise showed how often ties actually happen.

## Exercises

1. Add a `seed` input: while `push` is high and a new `seed_load` input is
   asserted, load the LFSR from `{token_in, ...}`. Mirror it in Python.
2. Implement **top-2 sampling**: track the best *two* logits and pick between
   them with one LFSR bit. Compare text quality with uniform noise.
3. Replace uniform noise with a crude Gumbel approximation: a small ROM mapping
   4 random bits to `-log(-log(u))` scaled to integers. Does the text improve?

Next: [Lesson 8: Tiny Tapeout](08_tiny_tapeout.md)
