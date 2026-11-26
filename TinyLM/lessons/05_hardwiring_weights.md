# Lesson 5: Hard-wiring the weights

## Two ways to store weights on a chip

1. **Memory (SRAM, DRAM, flash).** Weights are data. You can change the model,
   but every inference has to *move* the weights from memory to the
   arithmetic. On GPUs, that data movement is the bottleneck and most of the
   energy.
2. **Logic (ROM, hard-wired).** Weights are part of the circuit. Nothing is
   moved; the "weight" is which wire connects to which adder. You can never
   change the model, but it's dense, fast and very energy-efficient. This is
   the Taalas approach.

TinyLM uses option 2.

## A ROM is just a `case` statement

`model/gen_verilog.py` writes `rtl/generated/tinylm_weights.v`:

```verilog
// Layer-1 ternary weights W1[j][i], address = j*ND + i
module tinylm_w1_rom (input wire [7:0] addr, output reg [1:0] data);
    always @* begin
        case (addr)
            8'd0: data = 2'b11;      // -1
            8'd2: data = 2'b01;      // +1
            ...
            default: data = 2'd0;    // every zero weight lands here: free
        endcase
    end
endmodule
```

Synthesis turns this `case` into a block of gates: a Boolean function from
address bits to data bits. Things to notice:
- **Zero weights don't appear at all.** They fall into `default`. Sparse
  models need less logic.
- The synthesiser minimises the logic, so patterns in the weights get shared
  automatically.
- No memory compiler, no SRAM macro. This is just standard cells, which is
  exactly what Tiny Tapeout supports.

The generator also writes:
- `tinylm_params.vh`: the config (N, D, H, widths) as `localparam`s, pulled
  into the hand-written modules with `` `include ``;
- `tinylm_charmap.v`: ASCII ↔ token conversion;
- `tinylm_core_parallel.v`: the fully-unrolled core (lesson 10).

## Code generation is a hardware superpower

Hand-writing 1,800 weights would be absurd. Production chip teams generate huge
amounts of Verilog from scripts (register maps, ROMs, lookup tables, whole
datapaths). The pattern is always the same:

```
trusted data (model/weights/tt.json) ──► generator script ──► Verilog ──► simulate against the same data
```

Read `gen_verilog.py`. It's short. `rom()` writes any ROM, and
`check_ranges()` proves no accumulator can overflow before any Verilog is
written.

## Inspect the result

```bash
make gen
less rtl/generated/tinylm_weights.v
make synth-tt        # generic gate count
```

## How this scales (and why Taalas is hard)

Hard-wiring 1,800 weights costs a few hundred gates. Hard-wiring 8 billion
needs a different trick: Taalas reportedly uses a dense mask-ROM-like structure
where the weight is set by a single metal or via layer, packed right next to
the multiply-accumulate logic. Changing models means new masks for only those
layers. Same principle as your `case` statement, very different engineering.

## Exercises

1. Count the non-zero entries in `tinylm_w2_rom`. Compare with the
   "non-zero" count printed by training.
2. Retrain with a different seed, run `make gen synth-tt`, and compare gate
   counts. Same architecture, different weights, different area: the weights
   *are* the circuit.
3. (Harder) Change `rom()` to emit `assign data = addr == 8'd0 ? ... : ...`
   chains instead of `case`. Does yosys produce a different gate count?

Next: [Lesson 6: The sequential core](06_sequential_core.md)
