# Lesson 8: Putting TinyLM on a Tiny Tapeout chip

## The Tiny Tapeout interface

Every Tiny Tapeout project is a module called `tt_um_<something>` with
exactly these ports:

```verilog
input  wire [7:0] ui_in,    // 8 dedicated inputs
output wire [7:0] uo_out,   // 8 dedicated outputs
input  wire [7:0] uio_in,   // 8 bidirectional pins: input value
output wire [7:0] uio_out,  //                       output value
output wire [7:0] uio_oe,   //                       1 = drive the pin (output)
input  wire       ena,      // high when your design is selected (can ignore)
input  wire       clk,
input  wire       rst_n     // active-low reset
```

`rtl/tt_um_tinylm.v` maps TinyLM onto them:

| Pin | Meaning |
|---|---|
| `ui_in[4:0]` | token to push |
| `ui_in[5]` | PUSH (rising edge) |
| `ui_in[6]` | START (rising edge) |
| `ui_in[7]` | NOISE: 1 = sample, 0 = greedy |
| `uo_out[4:0]` | predicted token |
| `uo_out[5]` | BUSY |
| `uo_out[6]` | VALID (prediction ready) |
| `uio_out[7:0]` | predicted character as **ASCII** |

## Crossing clock domains: synchronisers

The pins are driven by the RP2040 microcontroller on the demo board, which
runs on its own clock. A signal that changes at an arbitrary moment can catch
a flip-flop mid-transition, leaving it **metastable** (neither 0 nor 1) for a
while. The fix, `rtl/sync_edge.v`:

```verilog
always @(posedge clk) s <= {s[1:0], in};   // flop 1 may go metastable, flop 2 cleans it up
assign rise = s[1] & ~s[2];                // one-cycle pulse on a rising edge
```

The edge detector turns a "button press" of any length into exactly one
`push` or `start` pulse. `tb/tb_tt.v` deliberately drives pins at times that
don't line up with the clock (`#7`, `#97`) to exercise this.

Why can `ui_in[4:0]` (the token) skip synchronising? Because it's set
*before* PUSH rises and held until afterwards, so it's stable whenever the
synchronised edge samples it. That's a **handshake**: one synchronised
control signal guards a bundle of data.

## Check it

```bash
make test-tt       # drives the pins like the RP2040 would, 200 tokens, 0 errors
make synth-tt      # generic gate count
```

Yosys reports about **1,550 logic gates + ~205 flip-flops** for the `tt`
config. A Tiny Tapeout tile holds roughly 1,000 standard cells, and flip-flops
are big cells, so plan on **2–4 tiles** (`tiles: "2x2"` in `tt/info.yaml`).
The Tiny Tapeout GitHub Action runs the real place-and-route and reports the
true utilisation; adjust from there.

If it's too big for your budget, shrink the model (`H=12` or `D=2` in
`model/config.py`). If you have room, grow it.

## Submitting

1. Rename the module to `tt_um_<your_github_username>_tinylm` in
   `rtl/tt_um_tinylm.v` and `tt/info.yaml`. Fill in title and author.
2. Bundle everything into one file (includes inlined):
   ```bash
   make tt-bundle    # → build/tt_submission/{src/project.v, info.yaml, docs/info.md}
   ```
3. Create a new repo from the **current shuttle's Verilog template** (linked
   from tinytapeout.com; there are templates per shuttle, e.g. IHP or GF180).
   Copy `src/project.v` and `docs/info.md` in. Merge the fields of our
   `info.yaml` into the template's (the format changes between shuttles, so
   trust the template's version).
4. Enable GitHub Actions and push. The "gds" action hardens your design and
   shows a 3D view and the tile utilisation. The "test" action runs the
   template's cocotb test; adapt it from `tb/tb_tt.v`, or start with a smoke
   test that pushes "the " and checks VALID rises.
5. Submit on tinytapeout.com and pay for the tiles. Then wait for silicon
   (typically months).

## Talking to the chip (MicroPython on the demo board)

A sketch using the Tiny Tapeout MicroPython SDK. Check the SDK docs for the
exact API of your board revision; names have changed between versions.

```python
from ttboard.demoboard import DemoBoard
CHARS = " abcdefghijklmnopqrstuvwxyz.,'!?"

tt = DemoBoard.get()
tt.shuttle.tt_um_yourname_tinylm.enable()
tt.clock_project_PWM(10_000_000)          # 10 MHz → ~14k chars/s
tt.reset_project(True); tt.reset_project(False)

def pulse(bit, tok=0, noise=0):
    tt.ui_in.value = (noise << 7) | tok
    tt.ui_in.value = (noise << 7) | (1 << bit) | tok
    tt.ui_in.value = (noise << 7) | tok

for c in "the cat ":
    pulse(5, CHARS.index(c))              # PUSH

out = ""
for _ in range(80):
    pulse(6, noise=1)                     # START
    while not (tt.uo_out.value & 0x40):   # wait for VALID
        pass
    out += CHARS[tt.uo_out.value & 0x1F]
print(out)
```

## Exercises

1. Add a "generate N characters" mode so the RP2040 only has to pulse START
   once. Where should the counter live?
2. Use `uo_out[7]` as a heartbeat (toggle on every `done`) and check it with a
   logic analyser or an LED.
3. Add a cocotb test for the TT template that replays `tb/vectors/core_vectors.hex`.

Next: [Lesson 9: FPGA](09_fpga.md)
