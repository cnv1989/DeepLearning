# Lesson 2: Finite-state machines and testbenches

## FSMs

Every controller in this project is a finite-state machine (FSM): the
inference core, the UART receiver and the serial-terminal front end. Two
styles are common:

**Two-process style** (state register + combinational next-state). You likely
wrote this in VHDL. See `examples/03_fsm/seq_detect.v`:

```verilog
always @(posedge clk) state <= rst ? S0 : next;     // the register

always @* begin                                     // the logic
    next = state;                                   // default!
    case (state)
        S0: next = bit_in ? S1 : S0;
        ...
    endcase
end
```

**One-process style** (everything in one clocked block). This is what
`rtl/tinylm_core.v` and `rtl/uart_rx.v` use. It's more compact when the FSM
also updates counters and datapath registers:

```verilog
always @(posedge clk)
    case (state)
        S_IDLE: if (start) begin state <= S_L1; busy <= 1; end
        ...
    endcase
```

**Moore vs Mealy:** a Moore output depends only on the state (`found` in the
sequence detector). A Mealy output also depends on the inputs. Moore outputs
are glitch-free and easier to time. Mealy outputs react a cycle earlier.

### Exercise 2a

Run the sequence detector test (`make examples`). Then change it to detect
`1101`. Update `expect_found` in the testbench by hand first, *then* run it.
Writing the expected answer before seeing the output is the habit that
catches bugs.

## Testbenches

A testbench is a Verilog module with no ports. It makes a clock, drives the
inputs and checks the outputs. Things that only exist in simulation:

```verilog
always #10 clk = ~clk;                    // 50 MHz clock (with `timescale 1ns/1ps)
initial begin ... end                     // runs once, top to bottom
@(posedge clk);                           // wait for a clock edge
#1;                                       // wait 1 time unit (sample after the edge)
wait (done == 1);                         // wait for a condition
$display("x=%0d", x);                     // print
$readmemh("file.hex", mem);               // load test vectors
$dumpfile("x.vcd"); $dumpvars(0, tb);     // record waveforms
$finish;
```

### Three levels of testbench

1. **Eyeball:** print values or look at waves. Fine for 5 minutes, useless after that.
2. **Self-checking:** the testbench knows the right answer and prints
   PASS/FAIL (all the `examples/`).
3. **Golden-model vectors:** a separate, trusted model (here, Python)
   generates the expected outputs, and the testbench replays them. This is
   how real chips get verified, and it's what `tb/tb_core.v` does:

```
model/ref_model.py ──► tb/vectors/core_vectors.hex ──► tb/tb_core.v ──► PASS/FAIL
```

Each 16-bit vector word encodes one operation (push a token, or start an
inference and check the result). Open `model/gen_vectors.py` and
`tb/tb_core.v` side by side and match up the bit fields.

### Avoiding race conditions in testbenches

Drive DUT inputs with `<=` (or just after a clock edge), and sample outputs a
little after the edge (`@(posedge clk); #1;`). If the testbench and the design
both change signals at exactly the edge, the result depends on simulator
event ordering. That's a race.

### Waveforms

```bash
make wave            # writes build/tb_core.vcd
gtkwave build/tb_core.vcd
```

Add `dut.state`, `dut.j`, `dut.k`, `dut.acc`, `dut.w1_addr` and `dut.done`.
You'll be able to watch the core step through the neural network one weight
per clock. This will make much more sense after lesson 6.

## Exercises

1. Break `rtl/tinylm_core.v` on purpose: change the `>` in
   `logit > best_val` to `>=` (so the last maximum wins instead of the first). Run
   `make test-core` and guess how many of the 200 tokens mismatch before you
   look. (I got 141. Small integer logits tie far more often than float
   logits would, and one wrong character feeds back into the context, so
   every later prediction diverges too.) Put it back.
2. Add a check to `tb/tb_core.v` that `busy` is high the whole time between
   `start` and `done`.

Next: [Lesson 3: Fixed-point arithmetic](03_fixed_point.md)
