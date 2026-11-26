# Lesson 6: The sequential inference core (`rtl/tinylm_core.v`)

## Area vs time

The network needs 192 + 512 = 704 multiply-accumulates per character. You can
build 704 adders and do them all at once (lesson 10), or build **one** adder
and reuse it 704 times. On a Tiny Tapeout budget the choice is easy: one MAC
per clock.

## Block diagram

```
           ┌──────── ctx (3 × 5-bit tokens) ◄─── push / prediction
           │  p
           ▼
 {tok,d} ─► EMB ROM ─► e (int4) ─┐
                                 ├─► mux ─► x ─┐
 j ──────► h_mem[16] ─► h (u4) ──┘             │
                                               ▼
 w1_addr ─► W1 ROM ─┐                  ┌──► ±x / 0 ──┐
 w2_addr ─► W2 ROM ─┴─► mux ─► w ──────┘             ▼
                                         acc_base + prod ─► acc_next ─► acc
 j ─► B1 ROM ─┐                          ▲
 k ─► B2 ROM ─┴─► bias ─ first? ─────────┘
                                acc_next ─► requant ─► h_mem[j]   (end of layer-1 neuron)
                                acc_next + noise ─► argmax ─► token_out  (end of layer-2 logit)
```

One adder does everything. The muxes pick layer 1 (embedding × W1) or layer
2 (hidden × W2) operands depending on the FSM state.

## The FSM

```
          start                last MAC of last hidden neuron
  IDLE ──────────► L1 ───────────────────────────────────────► L2
   ▲   (H × N × D = 192 cycles)          (V × H = 512 cycles)   │
   └──────────────────── last MAC of last logit ────────────────┘
                         done=1, token_out, ctx ← prediction
```

### Layer 1, one MAC per clock

Counters `d` (embedding element), `p` (context position) and `j` (neuron)
nest like loops, while `w1_addr` simply counts 0…191:

```python
for j in range(H):            # neuron
    for p in range(N):        # context position
        for d in range(D):    # embedding element
            acc = (b1[j] if first else acc) + W1[w1_addr] * E[ctx[p]][d]
            w1_addr += 1
    h[j] = requant(acc)
```

**Trick 1: no wasted cycles for the bias.** On the first MAC of a neuron,
the adder adds the product to the *bias* instead of to `acc`
(`acc_base = first ? bias : acc`).

**Trick 2: no wasted cycles for the result.** On the last MAC, the result is
taken from `acc_next` (the combinational sum) and requantised in the same
cycle. That's why the latency is exactly 192 + 512 = 704 compute cycles.

**Trick 3: counters instead of multipliers.** The ROM address `j*ND + i` would
need a multiplier, so `w1_addr` is a separate counter that just increments.

### Layer 2

Same loop shape: `j` walks the 16 hidden values, `k` walks the 32 logits,
`w2_addr` counts 0…511. At the end of each logit the core adds optional noise,
compares with the best so far (running argmax), and advances the LFSR
(lesson 7). After logit 31, the winner goes to `token_out`, is shifted into
`ctx`, and `done` pulses.

## Reading the code

Open `rtl/tinylm_core.v` next to this lesson. It's organised as:
1. State registers
2. ROM instances (the hard-wired weights)
3. **Datapath:** all combinational `wire`s: operand muxes, ternary product,
   `acc_next`, requantisation, noise, argmax compare
4. **Control:** one clocked `always` block, the FSM and counters

Separating combinational datapath from clocked control keeps the clocked block
readable: it only decides *what to store*, never how to compute it.

## Verify

```bash
make test-core
# tb_core: 200 tokens checked, 0 errors, 706 cycles/token
make wave && gtkwave build/tb_core.vcd
```

(706 = 704 compute cycles + 1 cycle to see `start` + 1 cycle for the
testbench to see `done`.)

## Exercises

1. **Pipeline the ROMs.** On a real chip, ROM → mux → adder in one cycle may
   limit the clock speed. Register the ROM outputs, which adds a pipeline
   stage. What changes in the FSM? Make the test pass again (latency will
   change; throughput shouldn't).
2. **Two MACs per cycle.** Process two weights per clock (two ROM reads, two
   ternary products, a 3-input add). Halve the latency and measure the area
   with `make synth-tt`.
3. **Skip zeros.** About 50% of weights are zero, yet the core still spends a
   cycle on each. Design a ROM that stores only non-zero (index, sign) pairs
   per neuron. What's the speed-up, and what does the variable latency do to
   the FSM?

Next: [Lesson 7: Argmax and sampling](07_sampling.md)
