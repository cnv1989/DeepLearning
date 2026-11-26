// Self-checking testbench for the TinyLM core.
// Replays tb/vectors/core_vectors.hex (made by model/gen_vectors.py from the
// bit-exact Python model) and checks every predicted token.
//
//   iverilog -g2012 -I rtl/generated -o build/tb_core.vvp tb/tb_core.v rtl/tinylm_core.v rtl/generated/tinylm_weights.v
//   vvp build/tb_core.vvp
// Add -DPARALLEL (and rtl/generated/tinylm_core_parallel.v) to test the parallel core.
`timescale 1ns/1ps
module tb_core;
    reg clk = 0;
    always #10 clk = ~clk;                  // 50 MHz

    reg        rst_n = 0, push = 0, start = 0, noise_en = 0;
    reg  [4:0] token_in = 0;
    wire       busy, done;
    wire [4:0] token_out;

`ifdef PARALLEL
    tinylm_core_parallel dut (
`else
    tinylm_core dut (
`endif
        .clk(clk), .rst_n(rst_n), .push(push), .token_in(token_in),
        .start(start), .noise_en(noise_en),
        .busy(busy), .done(done), .token_out(token_out));

`include "core_counts.vh"
    reg [15:0] vec [0:CORE_LEN-1];
    integer i, errors = 0, starts = 0, cycles, total_cycles = 0;

    initial begin
        if ($test$plusargs("vcd")) begin
            $dumpfile("build/tb_core.vcd");
            $dumpvars(0, tb_core);
        end
        $readmemh("tb/vectors/core_vectors.hex", vec);
        repeat (3) @(posedge clk);
        rst_n <= 1;
        @(posedge clk);

        for (i = 0; vec[i][15:14] != 2'd3; i = i + 1) begin
            if (vec[i][15:14] == 2'd0) begin                // push
                token_in <= vec[i][12:8];
                push     <= 1;
                @(posedge clk);
                push     <= 0;
            end else begin                                  // start + check
                noise_en <= vec[i][13];
                start    <= 1;
                @(posedge clk);
                start    <= 0;
                cycles = 1;
                while (!done) begin
                    @(posedge clk);
                    cycles = cycles + 1;
                end
                total_cycles = total_cycles + cycles;
                starts = starts + 1;
                if (token_out !== vec[i][4:0]) begin
                    errors = errors + 1;
                    $display("MISMATCH at vector %0d: got %0d expected %0d", i, token_out, vec[i][4:0]);
                end
            end
        end
        $display("tb_core: %0d tokens checked, %0d errors, %0d cycles/token",
                 starts, errors, total_cycles / starts);
        if (errors == 0) $display("PASS");
        else             $display("FAIL");
        $finish;
    end

    initial begin
        #200_000_000;
        $display("FAIL: timeout");
        $finish;
    end
endmodule
