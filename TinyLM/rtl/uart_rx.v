// UART receiver, 8N1. Samples each bit in the middle of its bit period.
// `valid` pulses for one cycle with the received byte on `data`.
module uart_rx #(
    parameter integer CLKS_PER_BIT = 104
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,
    output reg  [7:0] data,
    output reg        valid
);
    localparam [1:0] IDLE = 0, START = 1, BITS = 2, STOP = 3;
    reg [1:0] state;
    reg [$clog2(CLKS_PER_BIT)-1:0] cnt;
    reg [2:0] bitn;
    reg [1:0] rx_s;                   // synchroniser: rx is asynchronous!
    wire      r = rx_s[1];

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE; cnt <= 0; bitn <= 0; data <= 0; valid <= 0; rx_s <= 2'b11;
        end else begin
            rx_s  <= {rx_s[0], rx};
            valid <= 1'b0;
            case (state)
            IDLE:  if (!r) begin state <= START; cnt <= 0; end
            START: if (cnt == CLKS_PER_BIT/2 - 1) begin     // middle of start bit
                       cnt   <= 0;
                       state <= r ? IDLE : BITS;             // glitch -> back to idle
                       bitn  <= 0;
                   end else cnt <= cnt + 1'b1;
            BITS:  if (cnt == CLKS_PER_BIT - 1) begin
                       cnt  <= 0;
                       data <= {r, data[7:1]};               // LSB first
                       bitn <= bitn + 1'b1;
                       if (bitn == 3'd7) state <= STOP;
                   end else cnt <= cnt + 1'b1;
            STOP:  if (cnt == CLKS_PER_BIT - 1) begin
                       state <= IDLE;
                       valid <= r;                           // drop frames with a bad stop bit
                   end else cnt <= cnt + 1'b1;
            endcase
        end
    end
endmodule
