# TinyLM: a language model in silicon

A hands-on course that takes you from "I did some VHDL/Verilog at university"
to a working **character-level language model with its weights hard-wired
into logic**. The same idea as Taalas-style "model-as-silicon" chips, shrunk to
fit a few [Tiny Tapeout](https://tinytapeout.com) tiles or a $70 FPGA.

```
"the cat " ──► [ embed ] ──► [ 16 neurons ] ──► [ 32 logits ] ──► argmax ──► "s"
                 4-bit         ternary W1         ternary W2       + LFSR
                 ROM           -1/0/+1            -1/0/+1          sampling
```

Everything in this repo has been run and tested. The RTL matches the Python
reference model **bit for bit** on 200 generated characters, through the bare
core, the Tiny Tapeout pin interface and a full UART conversation.

| | Tiny Tapeout config (`tt`) | FPGA config (`fpga`) |
|---|---|---|
| Context | 3 characters | 6 characters |
| Embedding | 32 × 4 × int4 | 32 × 8 × int4 |
| Hidden layer | 16 neurons, ternary | 48 neurons, ternary |
| Next-char accuracy (training text) | ~38% | ~53% |
| Cycles per character | 706 | 3,842 |
| Speed @ 50 MHz (TT) / 12 MHz (iCE40) | ~70k chars/s | ~3k chars/s |
| Size | ~1.5k gates + ~200 flops | 987 LUTs on iCE40UP5K |

## The course

| # | Lesson | You build |
|---|---|---|
| 0 | [Setup](lessons/00_setup.md) | toolchain, run every test |
| 1 | [Verilog refresher (for VHDL people)](lessons/01_verilog_refresher.md) | counter, blocking vs non-blocking |
| 2 | [FSMs and testbenches](lessons/02_fsm_and_testbenches.md) | sequence detector, self-checking tests |
| 3 | [Fixed-point arithmetic for neural nets](lessons/03_fixed_point.md) | signed maths, ternary MAC |
| 4 | [The model: a tiny character LM](lessons/04_the_model.md) | quantisation-aware training in NumPy |
| 5 | [Hard-wiring the weights](lessons/05_hardwiring_weights.md) | weights → Verilog generator |
| 6 | [The sequential inference core](lessons/06_sequential_core.md) | `tinylm_core.v` |
| 7 | [Argmax and sampling](lessons/07_sampling.md) | LFSR noise, bit-exact randomness |
| 8 | [Tiny Tapeout](lessons/08_tiny_tapeout.md) | pin interface, synchronisers, submission |
| 9 | [FPGA](lessons/09_fpga.md) | UART, iCEBreaker bitstream |
| 10 | [Going parallel and scaling up](lessons/10_parallel_and_scaling.md) | one-token-per-clock "Taalas-style" core |

## Quick start

```bash
cd TinyLM
make test        # 4 testbenches: core, parallel core, Tiny Tapeout top, UART top
make examples    # refresher exercises
```

Retrain and regenerate everything (about 5 seconds):

```bash
make train gen test
```

## Layout

```
model/        Python: charset, configs, training, bit-exact reference, generators
  data/       training text (swap in your own)
rtl/          hand-written Verilog: core, UART, synchroniser, TT and FPGA tops
  generated/  weights-as-logic, produced by model/gen_verilog.py
tb/           self-checking testbenches + generated vectors
examples/     refresher exercises (lessons 1-3)
fpga/         iCEBreaker pin constraints
tt/           Tiny Tapeout info.yaml + datasheet page
tools/        bundle_tt.py: one-file Tiny Tapeout submission
```
