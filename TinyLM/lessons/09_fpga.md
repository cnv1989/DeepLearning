# Lesson 9: TinyLM on an FPGA

An FPGA lets you run the design on real hardware **today**, for the cost of a
dev board, and it's the step to take before paying for a tapeout.

## Target: iCEBreaker (Lattice iCE40UP5K)

A cheap, fully open-source-toolchain board: 5,280 LUTs, a 12 MHz clock, and a
USB-serial port. Other boards work too; see "Porting" below.

| Config | LUTs (of 5,280) | Cycles/char | Chars/s @ 12 MHz |
|---|---|---|---|
| `tt` | 766 | 706 | ~17,000 |
| `fpga` | 987 | 3,842 | ~3,100 |

(Totals include the UART front end.)

## The serial front end

`rtl/tinylm_uart_top.v` turns TinyLM into a chat-in-a-terminal:
- type characters: each one is mapped to a token, pushed into the context,
  and echoed;
- press **Enter**: TinyLM writes 64 characters;
- type **`~`**: toggle sampling noise (the green LED shows the mode).

It's built from:
- `rtl/uart_rx.v`: synchronises `rx` (asynchronous again), finds the start
  bit, samples each bit in the **middle** of its period, checks the stop bit;
- `rtl/uart_tx.v`: a 10-bit shift register `{stop, data, start}` clocked at
  the baud rate;
- a controller FSM with a small "subroutine" trick: state `S_SEND` sends
  `tx_data` and then jumps to the state stored in `after_send`, so one send
  routine is shared by echo, CR/LF and generated characters.

`CLKS_PER_BIT = 12 MHz / 115200 ≈ 104`. The real rate is 115,385 baud,
0.16% off, well inside UART tolerance (about ±2%).

## Simulate first

```bash
make test-uart
```

```
the cat
 the came came came came
~mary
 mint and the cam hed le
tb_uart_top: 68/68 bytes received, 0 errors
```

The testbench "types" into the serial line, decodes the design's output with
a second `uart_rx`, and compares every byte with what the Python model
predicts. It uses `CLKS_PER_BIT = 16` so simulation runs fast.

## Build and flash

You need the OSS CAD Suite (yosys, nextpnr-ice40, icepack, iceprog):

```bash
make bitstream          # yosys → nextpnr → icepack → build/tinylm.bin
make prog               # iceprog
```

Then open a terminal at 115200 8N1 (e.g. `screen /dev/ttyUSB1 115200` on Linux,
`screen /dev/tty.usbserial-* 115200` on macOS; the iCEBreaker exposes two
ports, usually the second one is the UART). Type `the cat` and press Enter.

To run the bigger model:

```bash
make train gen CONFIG=fpga
make test bitstream prog
```

(`make gen` overwrites `rtl/generated/`; run `make gen` again with
`CONFIG=tt` to switch back.)

Check `fpga/icebreaker.pcf` against your board's documentation. Pin numbers
are for iCEBreaker v1.0.

## Porting to other boards

Only `rtl/icebreaker_top.v` and the `.pcf` are board-specific. For another
board:
1. Write a small top that instantiates `tinylm_uart_top` with
   `CLKS_PER_BIT = clock_hz / 115200` and generates a reset.
2. Map clock, RX and TX in that board's constraints file (`.pcf` for iCE40,
   `.lpf` for ECP5, `.xdc` for Xilinx, `.cst` for Gowin).
3. Synthesise with that vendor's flow: yosys + nextpnr for iCE40, ECP5 and
   Gowin (Tang Nano); Vivado for Artix-7 (Arty, Basys 3).

On a larger FPGA (Artix-7, ECP5-85k) you can afford a much bigger model. Try
`H=128`, `N=8`, or the parallel core from lesson 10.

## Exercises

1. Add a 16-byte receive FIFO so you can paste a whole line while the model
   is generating.
2. Let the user type a number before Enter to set how many characters to
   generate.
3. Measure the real chars/s: count `done` pulses for one second with a timer
   and print the number over UART.

Next: [Lesson 10: Going parallel and scaling up](10_parallel_and_scaling.md)
