// Closed-loop replay test for TinyPilot.
// Each line of tb/vectors/flight.hex is one 1 kHz control loop recorded while
// the bit-exact Python model (model/pilot.py) flew the simulated quad:
//   [135:32] the 13-byte SPI frame sent to the chip, [31:0] expected motors.
// The testbench acts as the flight controller's MCU: it sends each frame over
// SPI and checks the motors read back in the NEXT frame. Finally it checks
// the Oneshot125 pulse width on motor 0.
`timescale 1ns/1ps
module tb_pilot;
    localparam integer HALF_SCK = 4;              // SCK = clk / 8
    localparam integer PERIOD = 2000, BASE = 500, STEP = 2;   // fast PWM for simulation

    reg clk = 0;
    always #10 clk = ~clk;
    reg rst_n = 0, sck = 0, mosi = 0, cs_n = 1, arm = 0;
    wire [7:0] uo_out, uio_out, uio_oe;

    tt_um_tinypilot #(.PWM_PERIOD(PERIOD), .PWM_BASE(BASE), .PWM_STEP(STEP)) dut (
        .ui_in({4'b0, arm, cs_n, mosi, sck}), .uo_out(uo_out), .uio_in(8'h00),
        .uio_out(uio_out), .uio_oe(uio_oe), .ena(1'b1), .clk(clk), .rst_n(rst_n));

`include "flight_len.vh"
    reg [135:0] vec [0:FLIGHT_LEN-1];
    reg [31:0]  got;
    integer t, b, errors = 0, max_busy = 0, cyc;

    // Full-duplex SPI transfer: send nbytes from `data` (MSB first), capture first 32 MISO bits.
    task spi(input [103:0] data, input integer nbytes);
        integer n;
        begin
            cs_n = 0; repeat (HALF_SCK) @(posedge clk);
            for (n = 0; n < nbytes * 8; n = n + 1) begin
                mosi = data[103 - n];
                repeat (HALF_SCK) @(posedge clk);
                sck = 1;
                if (n < 32) got[31 - n] = uo_out[0];
                repeat (HALF_SCK) @(posedge clk);
                sck = 0;
            end
            repeat (HALF_SCK) @(posedge clk);
            cs_n = 1;
            repeat (8 * HALF_SCK) @(posedge clk);
        end
    endtask

    task check(input integer idx);
        begin
            if (got !== vec[idx][31:0]) begin
                errors = errors + 1;
                if (errors < 10) $display("loop %0d: motors %h expected %h", idx, got, vec[idx][31:0]);
            end
        end
    endtask

    initial begin
        $readmemh("tb/vectors/flight.hex", vec);
        repeat (5) @(posedge clk); rst_n = 1; arm = 1;
        repeat (5) @(posedge clk);
        for (t = 0; t < FLIGHT_LEN; t = t + 1) begin
            spi(vec[t][135:32], 13);
            if (t > 0) check(t - 1);
            cyc = 0;                                   // wait for the inference
            @(posedge clk);
            while (uo_out[1]) begin @(posedge clk); cyc = cyc + 1; end
            if (cyc > max_busy) max_busy = cyc;
        end
        spi(104'd0, 4);                                // 4-byte read: no new inference
        check(FLIGHT_LEN - 1);

        // Oneshot125 check on motor 0: high time must be BASE + STEP * m0
        @(negedge uo_out[4]); @(posedge uo_out[4]);
        cyc = 0;
        while (uo_out[4]) begin @(posedge clk); cyc = cyc + 1; end
        $display("motor0 = %0d -> pulse %0d clocks (expected %0d)",
                 got[31:24], cyc, BASE + STEP * got[31:24]);
        if (cyc < BASE + STEP * got[31:24] - 1 || cyc > BASE + STEP * got[31:24] + 1) errors = errors + 1;

        $display("tb_pilot: %0d control loops checked, %0d errors, inference %0d cycles",
                 FLIGHT_LEN, errors, max_busy);
        if (errors == 0) $display("PASS"); else $display("FAIL");
        $finish;
    end
    initial begin #400_000_000; $display("FAIL: timeout"); $finish; end
endmodule
