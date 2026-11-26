// Lesson 1: blocking (=) vs non-blocking (<=). Run it and read the output.
// Rule of thumb: <= in clocked always blocks, = in combinational always @* blocks.
`timescale 1ns/1ps
module tb_blocking;
    reg clk = 0;
    always #5 clk = ~clk;

    reg [7:0] a_nb = 1, b_nb = 2;     // swapped with non-blocking
    reg [7:0] a_b  = 1, b_b  = 2;     // "swapped" with blocking

    always @(posedge clk) begin
        a_nb <= b_nb;                 // both right-hand sides are read BEFORE
        b_nb <= a_nb;                 // either register updates -> a real swap
    end
    always @(posedge clk) begin
        a_b = b_b;                    // a_b becomes 2 immediately ...
        b_b = a_b;                    // ... so b_b reads the NEW a_b -> both 2
    end

    // A 3-stage shift register written both ways:
    reg in = 0;
    reg [2:0] sr_nb = 0, sr_b = 0;
    always @(posedge clk) begin
        sr_nb[0] <= in; sr_nb[1] <= sr_nb[0]; sr_nb[2] <= sr_nb[1];   // 3 flops
    end
    always @(posedge clk) begin
        sr_b[0] = in; sr_b[1] = sr_b[0]; sr_b[2] = sr_b[1];            // collapses to 1 flop!
    end

    reg ok;
    initial begin
        @(posedge clk); #1;
        ok = (a_nb == 2 && b_nb == 1);
        $display("non-blocking swap: a=%0d b=%0d   (expected 2 1)", a_nb, b_nb);
        $display("blocking     swap: a=%0d b=%0d   (bug: 2 2)", a_b, b_b);
        in = 1;
        @(posedge clk); #1;
        $display("after one clock with in=1: shift(<=)=%b  shift(=)=%b  (expected 001 vs bug 111)", sr_nb, sr_b);
        $display("tb_blocking: %s", (ok && sr_nb == 3'b001) ? "PASS" : "FAIL");
        $finish;
    end
endmodule
