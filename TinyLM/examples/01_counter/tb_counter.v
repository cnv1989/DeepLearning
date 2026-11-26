`timescale 1ns/1ps
module tb_counter;
    reg clk = 0, rst = 1, en = 0;
    wire [3:0] q;
    wire wrap;
    counter #(.WIDTH(4)) dut (.clk(clk), .rst(rst), .en(en), .q(q), .wrap(wrap));

    always #5 clk = ~clk;            // 100 MHz clock

    integer errors = 0, i;
    initial begin
        $dumpfile("build/tb_counter.vcd");   // open with: gtkwave build/tb_counter.vcd
        $dumpvars(0, tb_counter);
        @(posedge clk); rst <= 0; en <= 1;
        for (i = 1; i <= 20; i = i + 1) begin
            @(posedge clk); #1;              // sample just after the edge
            if (q !== i[3:0]) begin
                errors = errors + 1;
                $display("cycle %0d: q=%0d expected %0d", i, q, i[3:0]);
            end
        end
        en <= 0; @(posedge clk); @(posedge clk); #1;
        if (q !== 4'd4) errors = errors + 1;     // 20 mod 16 = 4, held while en=0
        $display("tb_counter: %s", errors ? "FAIL" : "PASS");
        $finish;
    end
endmodule
