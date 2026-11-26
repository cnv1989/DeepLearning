# Lesson 3: Fixed-point arithmetic for neural networks

Floating point is expensive in silicon. TinyLM uses only small integers:

| Quantity | Type | Range |
|---|---|---|
| Embedding values `E` | signed 4-bit | −7 … 7 |
| Weights `W1`, `W2` | **ternary** | −1, 0, +1 |
| Biases `b1`, `b2` | signed 10-bit | −511 … 511 |
| Hidden activations `h` | unsigned 4-bit | 0 … 15 |
| Accumulators | signed 16-bit | −32768 … 32767 |

## Two's complement in 60 seconds

For a 4-bit signed number the bits `1101` mean −8 + 4 + 1 = **−3**. To widen
a signed number, copy the sign bit into the new upper bits:
`1101` → `1111_1101` (still −3). This is **sign extension**. Widening an
unsigned number pads with zeros instead (`1101` → `0000_1101` = 13).

## Verilog signed gotchas (run `examples/04_signed/tb_signed.v`)

1. **One unsigned operand makes the whole expression unsigned.**
   `e + 4'd1` with `e` signed is evaluated *unsigned*. Always use
   `4'sd1` or `$signed(...)`.
2. **Concatenation `{...}` is always unsigned.** Wrap it: `$signed({4'd0, h})`.
3. **`>>` is a logical shift, `>>>` is arithmetic** (and only when the operand
   is signed). Dividing a negative number needs `>>>`.
4. `>>> 1` rounds toward −∞ (floor). For round-half-up, add half first:
   `(x + 2) >>> 2` ≈ round(x / 4).
5. Assigning a signed value to a wider signed `wire` sign-extends it. That's
   the clean way: `wire signed [15:0] x1 = e_val;`.

## Ternary weights: multipliers disappear

With weights in {−1, 0, +1}, `w * x` is just a 3-way choice:

```verilog
wire signed [15:0] prod = (w == 2'b01) ?  x :      // +1
                          (w == 2'b11) ? -x :      // -1
                                          16'sd0;  //  0
```

There's no multiplier, so a whole MAC unit is one adder and a mux. Read
`examples/05_mac/ternary_mac.v`, which computes one neuron and checks it
against plain integer maths. This is the same idea behind BitNet b1.58
(ternary LLMs) and why ternary models suit hardware so well.

## Requantisation: keeping numbers small between layers

The hidden layer's accumulator can be large (±100 or so), but the next layer
wants 4-bit inputs. TinyLM squeezes it back down:

```
h = clamp( round( relu(acc) / 2^S1 ), 0, 15 )
```

In hardware that's a ReLU (sign check), a shift with rounding, and a
saturation (clamp). It's the same in Python (`requant` in
`model/ref_model.py`) and in Verilog (`h_new` in `rtl/tinylm_core.v`).
Compare the two; they must agree exactly.

## Proving no overflow

The core uses a 16-bit accumulator. Will it overflow? Don't guess; work out
the worst case. `model/gen_verilog.py` (`check_ranges`) does this for every
neuron:

```
worst = |bias| + (number of non-zero weights) × (largest |input|)
```

If any neuron's worst case doesn't fit in `ACC_W` bits, generation fails.
Worst-case bounds like this are how hardware engineers sleep at night.

## Exercises

1. In `examples/05_mac/tb_ternary_mac.v`, make `x` 8 bits wide and run 300
   accumulations. When does a 16-bit accumulator overflow? Add a check.
2. Work out the worst-case layer-1 accumulator for the `tt` config by hand:
   12 inputs, |x| ≤ 8, |b1| ≤ 511. How many bits does it actually need?
3. Change `requant` in `model/ref_model.py` to round half **down**
   (`(acc + (1 << (s1 - 1)) - 1) >> s1`), then run `make gen test-core`.
   Predict the result first. The vectors now come from the changed Python,
   but the Verilog didn't change, so any neuron that lands exactly on a .5
   mismatches. This is the drift a golden model catches. Put it back
   afterwards.

Next: [Lesson 4: The model](04_the_model.md)
