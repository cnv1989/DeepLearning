// Tiny Tapeout top for TinyPilot. Rename to tt_um_<github_username>_tinypilot.
//
//   ui_in[0] SCK   ui_in[1] MOSI   ui_in[2] CS_N   ui_in[3] ARM (1 = motors may spin)
//   uo_out[0] MISO   uo_out[1] BUSY   uo_out[2] ARMED   uo_out[3] 0
//   uo_out[7:4] Oneshot125 ESC signals for motors 3..0 (uo_out[4] = motor 0)
`default_nettype none
module tt_um_tinypilot #(
    parameter integer PWM_PERIOD = 25000,
    parameter integer PWM_BASE   = 6250,
    parameter integer PWM_STEP   = 24
) (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);
    wire [103:0] frame;
    wire         frame_valid, busy, done, miso;
    wire [31:0]  motors;

    spi_slave u_spi (.clk(clk), .rst_n(rst_n), .sck(ui_in[0]), .mosi(ui_in[1]), .cs_n(ui_in[2]),
                     .miso(miso), .tx_word(motors), .frame(frame), .frame_valid(frame_valid));

    pilot_core u_core (.clk(clk), .rst_n(rst_n), .start(frame_valid),
                       .sp(frame[103:56]), .gyro(frame[55:8]), .thr(frame[7:0]),
                       .busy(busy), .done(done), .motors(motors));

    reg [1:0] arm_s;
    always @(posedge clk) arm_s <= rst_n ? {arm_s[0], ui_in[3]} : 2'b00;

    wire [3:0] pwm;
    oneshot125 #(.PERIOD(PWM_PERIOD), .BASE(PWM_BASE), .STEP(PWM_STEP)) u_pwm (
        .clk(clk), .rst_n(rst_n), .armed(arm_s[1]), .duty(motors), .pwm(pwm));

    assign uo_out  = {pwm[3], pwm[2], pwm[1], pwm[0], 1'b0, arm_s[1], busy, miso};
    assign uio_out = 8'h00;
    assign uio_oe  = 8'h00;
    wire _unused = &{ena, uio_in, done, 1'b0};
endmodule
`default_nettype wire
