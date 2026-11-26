// Lesson 3: signed arithmetic gotchas. Every line prints what Verilog does.
`timescale 1ns/1ps
module tb_signed;
    reg  signed [3:0] e;          // a 4-bit embedding value, range -8..7
    reg         [3:0] u;
    reg  signed [15:0] acc;
    integer errors = 0;

    task check(input [8*48-1:0] what, input integer got, input integer want);
        begin
            $display("%0s = %0d (expected %0d)%0s", what, got, want, got == want ? "" : "  <-- WRONG");
            if (got != want) errors = errors + 1;
        end
    endtask

    initial begin
        e = -4'sd3;  u = 4'd13;                      // same bits: 1101
        acc = e;                 check("signed 4b -> signed 16b (sign-extends)", acc, -3);
        acc = u;                 check("unsigned 4b -> 16b (zero-extends)", acc, 13);
        acc = {12'd0, e};        check("concatenation is ALWAYS unsigned", acc, 13);
        acc = $signed({{12{e[3]}}, e}); check("manual sign extension", acc, -3);
        acc = e + 4'd1;          check("signed + unsigned -> whole expr unsigned!", acc, 14);
        acc = e + 4'sd1;         check("signed + signed", acc, -2);
        acc = -16'sd7 >>> 1;     check("-7 >>> 1 (arithmetic shift = floor)", acc, -4);
        acc = -16'sd7 >> 1;      check("-7 >> 1 (logical shift, garbage)", acc, 32764);
        acc = (16'sd7 + 16'sd2) >>> 2; check("round(7/4) as (x+2)>>>2", acc, 2);
        $display("tb_signed: %s", errors ? "FAIL" : "PASS");
        $finish;
    end
endmodule
