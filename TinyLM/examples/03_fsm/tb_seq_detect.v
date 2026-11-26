`timescale 1ns/1ps
module tb_seq_detect;
    reg clk = 0, rst = 1, bit_in = 0;
    wire found;
    seq_detect dut (.clk(clk), .rst(rst), .bit_in(bit_in), .found(found));
    always #5 clk = ~clk;

    // stream:         1 0 1 1 0 1 1 1 0 1 0 1 1
    reg [12:0] stream = 13'b1011011101011;
    // found is high the cycle after the last bit of each 1011 (overlap allowed):
    reg [12:0] expect_found = 13'b0001001000001;
    integer i, errors = 0;
    initial begin
        @(posedge clk); rst <= 0;
        for (i = 12; i >= 0; i = i - 1) begin
            bit_in <= stream[i];
            @(posedge clk); #1;
            if (found !== expect_found[i]) begin
                errors = errors + 1;
                $display("bit %0d: found=%b expected %b", 12 - i, found, expect_found[i]);
            end
        end
        $display("tb_seq_detect: %s", errors ? "FAIL" : "PASS");
        $finish;
    end
endmodule
