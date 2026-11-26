# Lesson 10: Going parallel ("Taalas-style") and scaling up

## The fully-parallel core

`rtl/generated/tinylm_core_parallel.v` has the same ports as the sequential
core, but **the entire network is one combinational circuit**: start → done
in one clock. Because the weights are ternary constants, every neuron becomes
a fixed adder tree of exactly the wires it needs:

```verilog
wire signed [15:0] a3 = 16'sd0 - 16'sd4 + x0 - x2 + x5 + x6 - x9 - x11;   // neuron 3
wire [3:0] h3 = requant(a3);
wire signed [15:0] z7 = 16'sd0 + 16'sd12 + h0 - h3 + h4 + ... ;            // logit 7
```

No ROMs, no multipliers, no counters. **The weights have become the wiring.**
Zero weights vanish completely. This is the core idea behind hard-wired LLM
chips, shown at the smallest possible scale.

It passes the same bit-exact test, including the LFSR noise (32 LFSR steps
are unrolled combinationally):

```bash
make test-parallel
# tb_core: 200 tokens checked, 0 errors, 2 cycles/token
```

## The trade-off, measured

Both versions of the Tiny Tapeout top, mapped to iCE40 LUTs (`make synth-compare`):

| | Sequential | Parallel |
|---|---|---|
| LUTs | 594 | 5,225 |
| Cycles per character | 704 | 1 |
| Relative area | 1× | ~9× |
| Relative throughput (same clock) | 1× | ~700× |

The parallel version needs about 9× the logic for about 700× the work per
clock, because it doesn't spend area on control, ROM addressing or muxes. Its
long adder and argmax chains run at a lower clock, but it's still hundreds of
times faster. On an iCE40UP5K it only just fits (5,225 of 5,280 LUTs); on a
Tiny Tapeout chip it would need a lot of tiles.

This is the trade-off Taalas makes at the extreme: spend silicon area to
eliminate data movement and control, and get 10–100× the tokens per second of
a GPU for one fixed model.

### Middle ground

Real designs live between the extremes:
- **Neuron-parallel:** compute all 16 hidden neurons at once (16 adders),
  looping over inputs: 12 + 32 cycles instead of 704.
- **Pipelined parallel:** add registers between layer 1 and layer 2 so the
  clock runs faster, and one token per clock is still possible when batching
  several independent streams.
- **Constant-coefficient folding:** with fixed weights, synthesis can share
  partial sums between neurons that use the same inputs with the same signs.

## Scaling up: what changes for a real LLM

| | TinyLM | Llama-class 8B |
|---|---|---|
| Parameters | ~900 | 8,000,000,000 |
| Weight storage | `case` statements → gates | dense mask ROM next to compute |
| Context | fixed 3-char window | thousands of tokens |
| Memory needed at run time | 15 bits of context | **KV cache**: gigabytes of SRAM/DRAM |
| Layers | 2 | ~32 transformer blocks |

The big missing piece is **attention**. A transformer must remember keys and
values for every previous token, which is read/write memory that grows with
the context. You can't hard-wire it. Even Taalas-style chips need real memory
for the KV cache; only the *weights* are frozen.

### A transformer on an FPGA: the next project

A good step up from this course is a one-layer, character-level transformer
on a mid-size FPGA:
- `d_model = 32`, one attention head, context 32, ternary weights;
- Q/K/V projections: reuse the sequential MAC core with three weight ROMs;
- KV cache: two small block RAMs (32 × 32 × 8 bits each);
- softmax: a 256-entry `exp` lookup table plus a divider, or the
  "argmax of scores" approximation;
- verify it the same way: a NumPy integer golden model plus generated vectors.

### Real silicon budgets (rough, late 2026)

| Target | Model you can hard-wire | Cost |
|---|---|---|
| Tiny Tapeout, 2–8 tiles | this course (~1k params) | a few hundred € |
| Full GF180 die (wafer.space) | ~50k–500k ternary params, small MLP or tiny transformer | ~$7k–15k |
| 28nm MPW | ~10M–100M params | $200k+ |
| 6nm full masks | billions (Taalas) | $50M+ |

## Where to go from here

1. **Ship it:** get TinyLM running on an FPGA this week (lesson 9). Submit to
   the next Tiny Tapeout shuttle (lesson 8).
2. **Make it better:** add a third layer, a bigger context, or skip zero
   weights (lesson 6, exercise 3). Train on your own text.
3. **Learn the real flow:** the Tiny Tapeout GitHub Action runs OpenLane or
   LibreLane. Read its reports (timing, area, power) and find your critical path.
4. **Go further:** the transformer project above, then a GF180 die of your own.

Back to [the course index](../README.md)
