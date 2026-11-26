// Testbench for the Tiny Tapeout top: drives the pins the way the RP2040
// on the demo board would (slow, asynchronous pulses) and checks the
// predicted ASCII characters against the Python reference model.
`timescale 1ns/1ps
module tb_tt;
    reg clk = 0;
    always #10 clk = ~clk;

    reg  [7:0] ui_in = 0;
    reg        rst_n = 0;
    wire [7:0] uo_out, uio_out, uio_oe;

    tt_um_tinylm dut (.ui_in(ui_in), .uo_out(uo_out), .uio_in(8'h00), .uio_out(uio_out),
                      .uio_oe(uio_oe), .ena(1'b1), .clk(clk), .rst_n(rst_n));

`include "core_counts.vh"
    reg [15:0] vec [0:CORE_LEN-1];
    integer i, errors = 0, starts = 0;

    // A "GPIO" pulse from a microcontroller: not aligned to our clock, a few cycles long.
    task pulse(input integer bit_idx);
        begin
            #7;  ui_in[bit_idx] = 1'b1;
            #97; ui_in[bit_idx] = 1'b0;
            #53;
        end
    endtask

    initial begin
        $readmemh("tb/vectors/core_vectors.hex", vec);
        #100 rst_n = 1;
        #100;
        for (i = 0; vec[i][15:14] != 2'd3; i = i + 1) begin
            if (vec[i][15:14] == 2'd0) begin
                ui_in[4:0] = vec[i][12:8];
                pulse(5);                                   // PUSH
            end else begin
                ui_in[7] = vec[i][13];                      // NOISE level
                #60;
                pulse(6);                                   // START
                wait (uo_out[6] == 1'b1);                   // VALID
                #1;
                starts = starts + 1;
                if (uo_out[4:0] !== vec[i][4:0]) begin
                    errors = errors + 1;
                    $display("MISMATCH at vector %0d: got %0d expected %0d", i, uo_out[4:0], vec[i][4:0]);
                end
                $write("%c", uio_out);                     // the ASCII pins
            end
        end
        $display("\ntb_tt: %0d tokens checked, %0d errors, uio_oe=%h", starts, errors, uio_oe);
        if (errors == 0 && uio_oe == 8'hFF) $display("PASS");
        else                                $display("FAIL");
        $finish;
    end

    initial begin
        #500_000_000;
        $display("FAIL: timeout");
        $finish;
    end
endmodule
