// Four-channel Oneshot125 ESC output: a pulse of 125..250 us every PERIOD clocks.
// duty 0 -> 125 us (motor stop), 255 -> ~250 us (full). Not armed -> always 125 us.
// Duties are latched at the start of each period, so a pulse never glitches.
module oneshot125 #(
    parameter integer PERIOD = 25000,   // 500 us at 50 MHz (2 kHz update)
    parameter integer BASE   = 6250,    // 125 us at 50 MHz
    parameter integer STEP   = 24       // 255 * 24 clocks ~= 122 us
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        armed,
    input  wire [31:0] duty,            // {m0, m1, m2, m3}
    output reg  [3:0]  pwm
);
    reg [$clog2(PERIOD)-1:0] cnt;
    reg [31:0] latched;
    integer c;
    always @(posedge clk) begin
        if (!rst_n) begin
            cnt <= 0; latched <= 0; pwm <= 0;
        end else begin
            cnt <= (cnt == PERIOD - 1) ? 0 : cnt + 1'b1;
            if (cnt == PERIOD - 1) latched <= armed ? duty : 32'd0;
            for (c = 0; c < 4; c = c + 1)
                pwm[c] <= (cnt < BASE + STEP * latched[31 - 8*c -: 8]);
        end
    end
endmodule
