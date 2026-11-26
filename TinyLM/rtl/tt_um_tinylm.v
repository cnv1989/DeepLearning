// Tiny Tapeout top level for TinyLM.
// Rename the module to tt_um_<your_github_username>_tinylm before submitting
// (and update tt/info.yaml to match).
//
// Pin map (driven by the RP2040 on the TT demo board, or any microcontroller):
//   ui_in[4:0]  token to push
//   ui_in[5]    PUSH   rising edge -> push ui_in[4:0] into the context
//   ui_in[6]    START  rising edge -> predict the next character
//   ui_in[7]    NOISE  1 = sample (random), 0 = greedy (argmax)
//   uo_out[4:0] predicted token
//   uo_out[5]   BUSY
//   uo_out[6]   VALID  set when a prediction is ready, cleared by PUSH/START
//   uo_out[7]   0
//   uio_out     predicted character as ASCII (all 8 uio pins are outputs)
`default_nettype none
module tt_um_tinylm (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);
    // Inputs from the outside world are asynchronous -> synchronise them.
    wire push_rise, start_rise, noise_lvl;
    sync_edge u_push  (.clk(clk), .rst_n(rst_n), .in(ui_in[5]), .level(),          .rise(push_rise));
    sync_edge u_start (.clk(clk), .rst_n(rst_n), .in(ui_in[6]), .level(),          .rise(start_rise));
    sync_edge u_noise (.clk(clk), .rst_n(rst_n), .in(ui_in[7]), .level(noise_lvl), .rise());

    wire       busy, done;
    wire [4:0] token_out;
`ifdef TINYLM_PARALLEL
    tinylm_core_parallel u_core (
`else
    tinylm_core u_core (
`endif
        .clk(clk), .rst_n(rst_n),
        .push(push_rise), .token_in(ui_in[4:0]),
        .start(start_rise), .noise_en(noise_lvl),
        .busy(busy), .done(done), .token_out(token_out));

    reg valid;
    always @(posedge clk) begin
        if (!rst_n)                       valid <= 1'b0;
        else if (push_rise | start_rise)  valid <= 1'b0;
        else if (done)                    valid <= 1'b1;
    end

    wire [7:0] ch;
    tinylm_tok2char u_t2c (.tok(token_out), .ch(ch));

    assign uo_out  = {1'b0, valid, busy, token_out};
    assign uio_out = ch;
    assign uio_oe  = 8'hFF;

    wire _unused = &{ena, uio_in, 1'b0};
endmodule
`default_nettype wire
