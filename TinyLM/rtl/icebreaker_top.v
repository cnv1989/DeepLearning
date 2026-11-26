// iCEBreaker (Lattice iCE40UP5K, 12 MHz) wrapper. Serial: 115200 8N1 over the
// FTDI USB port. Green LED = noise/sampling on. See fpga/icebreaker.pcf.
module icebreaker_top (
    input  wire CLK,
    input  wire RX,
    output wire TX,
    output wire LEDG_N,
    output wire LEDR_N
);
    // Power-on reset: hold reset for 255 clocks after configuration.
    reg [7:0] por = 8'd0;
    always @(posedge CLK) if (!por[7]) por <= por + 1'b1;
    wire rst_n = por[7];

    wire noise_en;
    tinylm_uart_top #(.CLKS_PER_BIT(104), .GEN_LEN(64)) u_top (
        .clk(CLK), .rst_n(rst_n), .rx(RX), .tx(TX), .noise_en(noise_en));

    assign LEDG_N = ~noise_en;
    assign LEDR_N = rst_n;   // red while in reset
endmodule
