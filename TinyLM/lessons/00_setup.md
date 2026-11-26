# Lesson 0: Setup

## Tools

Everything is free and open source.

| Tool | What for | Install (Ubuntu/Debian) | macOS (Homebrew) |
|---|---|---|---|
| Icarus Verilog | simulator | `sudo apt install iverilog` | `brew install icarus-verilog` |
| Yosys | synthesis (area estimates, FPGA) | `sudo apt install yosys` | `brew install yosys` |
| GTKWave or Surfer | waveform viewer | `sudo apt install gtkwave` | `brew install --cask gtkwave` or [surfer-project.org](https://surfer-project.org) |
| Python 3 + NumPy | model training, generators | `pip install numpy` | same |
| nextpnr-ice40, icestorm | FPGA place & route, bitstream (lesson 9) | the [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) bundles all of these | same |

**Easiest option:** download the **OSS CAD Suite**. It gives you iverilog,
yosys, nextpnr, icestorm, verilator, gtkwave and cocotb in one folder.

Optional extras for later: **Verilator** (fast simulator and linter) and
**cocotb** (testbenches in Python, which the Tiny Tapeout template uses).

## Check that everything works

```bash
cd TinyLM
make examples     # 5 refresher exercises
make test         # 4 TinyLM testbenches
```

You should see `PASS` eight or nine times, including:

```
tb_core: 200 tokens checked, 0 errors, 706 cycles/token
tb_core: 200 tokens checked, 0 errors, 2 cycles/token
tb_tt: 200 tokens checked, 0 errors, uio_oe=ff
tb_uart_top: 68/68 bytes received, 0 errors
```

## How the pieces fit together

```
model/data/corpus.txt
        │  model/train.py          (quantisation-aware training, NumPy)
        ▼
model/weights/tt.json              (integers only: E, W1, b1, W2, b2)
        │                    \
        │ model/gen_verilog.py \ model/gen_vectors.py  (uses model/ref_model.py)
        ▼                       ▼
rtl/generated/*.v          tb/vectors/*.hex     (expected outputs)
        │                       │
        └──► rtl/*.v ──► tb/*.v ◄┘   iverilog: PASS / FAIL
                 │
                 ├──► yosys ──► Tiny Tapeout (GDS via the TT GitHub Action)
                 └──► yosys + nextpnr ──► FPGA bitstream
```

The most important idea in this course: **there is one bit-exact golden
model** (`model/ref_model.py`). Training is built to match it, and the
hardware is tested against it. When the hardware and the golden model agree on
every bit, you can trust the chip will behave the way the Python did.

Next: [Lesson 1: Verilog refresher](01_verilog_refresher.md)
