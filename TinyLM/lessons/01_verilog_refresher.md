# Lesson 1: Verilog refresher (for people who remember some VHDL)

## The one thing to remember

Verilog *looks* like C, but you are **describing circuits**, not writing a
program. Every `always` block and every `assign` runs in parallel, all the
time. Ask yourself "what gates and flip-flops does this make?" for every line
you write.

## VHDL → Verilog cheat sheet

| Concept | VHDL | Verilog |
|---|---|---|
| Design unit | `entity` + `architecture` | `module ... endmodule` |
| Generic | `generic (W : integer := 8)` | `#(parameter integer W = 8)` |
| Port | `a : in std_logic_vector(7 downto 0)` | `input wire [7:0] a` |
| Signal | `signal s : std_logic_vector(3 downto 0)` | `wire [3:0] s;` or `reg [3:0] s;` |
| Concurrent assignment | `y <= a and b;` | `assign y = a & b;` |
| Clocked process | `process(clk) ... if rising_edge(clk)` | `always @(posedge clk)` |
| Combinational process | `process(all)` | `always @*` |
| Signal assignment in process | `q <= d;` | `q <= d;` (non-blocking) |
| Variable assignment | `v := x;` | `v = x;` (blocking) |
| Concatenation | `a & b` | `{a, b}` |
| Replication | `(others => '0')` | `{W{1'b0}}` |
| Literal | `x"A5"`, `"1010"` | `8'hA5`, `4'b1010`, `5'd17` |
| Signed type | `signed(...)` from numeric_std | `reg signed [7:0] x;` / `$signed(x)` |
| Case | `case s is when ... =>` | `case (s) ... endcase` |
| Instantiation | `u: entity work.foo port map (a => x);` | `foo u (.a(x));` |
| Generate | `for i in ... generate` | `generate for (...) ... endgenerate` |
| Not equal | `/=` | `!=` |
| Bit slice | `v(7 downto 4)` | `v[7:4]`, or indexed `v[i*4 +: 4]` |

### `wire` vs `reg`

This is the classic confusion. **`reg` does not mean "register".**
- `wire`: driven by `assign` or by a module output.
- `reg`: assigned inside an `always` block. It becomes a flip-flop only if
  that `always` block is clocked.

(SystemVerilog's `logic` replaces both. The code here uses plain Verilog-2005
so it works with every open-source tool.)

## The two kinds of `always` block

```verilog
// 1. Sequential: flip-flops. Use NON-BLOCKING <=
always @(posedge clk) begin
    if (rst) q <= 0;
    else     q <= q + 1'b1;
end

// 2. Combinational: gates. Use BLOCKING =, and assign every output on every path
always @* begin
    y = 0;                 // default value: prevents an accidental latch
    if (sel) y = a;
end
```

**Golden rules:**
1. Clocked block → `<=`. Combinational block → `=`. Never mix them in one block.
2. In a combinational block, give every output a value on every path, or you
   infer a **latch**. Latches are almost always a bug, and Tiny Tapeout's flow
   will warn about them.
3. Drive each signal from exactly one `always` block or `assign`.

### Exercise 1a: counter

Read `examples/01_counter/counter.v` and its testbench, then run:

```bash
make examples
iverilog -g2012 -o build/c.vvp examples/01_counter/*.v && vvp build/c.vvp
gtkwave build/tb_counter.vcd      # or open it in Surfer
```

In the waveform, look at `q` and `wrap`. Notice that `wrap` is combinational:
it changes in the same cycle `q` reaches 15.

### Exercise 1b: blocking vs non-blocking

Run `examples/02_blocking/tb_blocking.v`. It shows two classic bugs:
- a "swap" written with `=` gives `a=2 b=2` instead of swapping;
- a 3-stage shift register written with `=` collapses into a single flip-flop.

**Why:** with `<=`, every right-hand side is evaluated first, then all the
registers update together at the clock edge. That is exactly what real
flip-flops do.

## Parameters and sized literals

```verilog
module counter #(parameter integer WIDTH = 4) (...);
counter #(.WIDTH(8)) u_cnt (...);     // override when instantiating
```

Always size your literals: `1'b1`, `4'd3`, `16'sd5` (signed). An unsized `1`
is a 32-bit integer and can quietly widen your expressions.

## Indexed part-select (used a lot in TinyLM)

```verilog
reg [14:0] ctx;                  // three 5-bit tokens packed together
wire [4:0] tok = ctx[p*5 +: 5];  // "5 bits starting at p*5", and p can be a signal
```

## Exercises

1. Change the counter to count down when a new input `dir` is 1. Extend the
   testbench to check it.
2. Add a `load` input that loads a parameterised value. What happens if `load`
   and `en` are both high? Decide on a priority and test it.
3. Write a combinational `always @*` block that forgets a default assignment,
   and run `yosys -p "read_verilog yourfile.v; proc"`. Find the latch warning.

Next: [Lesson 2: FSMs and testbenches](02_fsm_and_testbenches.md)
