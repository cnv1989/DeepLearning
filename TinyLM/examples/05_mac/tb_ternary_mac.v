// Computes one neuron: bias + sum(w_i * x_i), and checks it against a
// reference computed with ordinary integers inside the testbench.
`timescale 1ns/1ps
module tb_ternary_mac;
    reg clk = 0;
    always #5 clk = ~clk;
    reg clear = 0, en = 0;
    reg signed [15:0] bias = 16'sd5;
    reg [1:0] w;
    reg signed [3:0] x;
    wire signed [15:0] acc;
    ternary_mac dut (.clk(clk), .clear(clear), .en(en), .bias(bias), .w(w), .x(x), .acc(acc));

    integer i, wi, ref_sum, seed = 42;
    initial begin
        ref_sum = bias;
        for (i = 0; i < 12; i = i + 1) begin
            wi = ($urandom(seed) % 3) - 1;              // -1, 0 or +1
            seed = seed + 1;
            w  <= (wi == 1) ? 2'b01 : (wi == -1) ? 2'b11 : 2'b00;
            x  <= $random % 8;                         // -7..7
            #0;
            clear <= (i == 0); en <= 1;
            @(posedge clk); #1;
            ref_sum = ref_sum + wi * x;
        end
        en <= 0;
        $display("neuron = %0d, reference = %0d", acc, ref_sum);
        $display("tb_ternary_mac: %s", (acc == ref_sum) ? "PASS" : "FAIL");
        $finish;
    end
endmodule
