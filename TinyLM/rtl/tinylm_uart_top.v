// Board-independent FPGA top: talk to TinyLM over a serial terminal.
//   type characters  -> they are pushed into the context and echoed
//   press Enter      -> TinyLM writes GEN_LEN characters
//   type '~'         -> toggle sampling noise (greedy <-> random)
module tinylm_uart_top #(
    parameter integer CLKS_PER_BIT = 104,   // clk_hz / baud
    parameter integer GEN_LEN      = 64
) (
    input  wire clk,
    input  wire rst_n,
    input  wire rx,
    output wire tx,
    output reg  noise_en
);
    wire [7:0] rx_data;
    wire       rx_valid;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (.clk(clk), .rst_n(rst_n), .rx(rx), .data(rx_data), .valid(rx_valid));

    reg        tx_start;
    reg  [7:0] tx_data;
    wire       tx_busy, tx_done;
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (.clk(clk), .rst_n(rst_n), .start(tx_start), .data(tx_data),
                                                .tx(tx), .busy(tx_busy), .done(tx_done));

    reg        push, start;
    reg  [4:0] push_tok;
    wire       busy, done;
    wire [4:0] token_out;
`ifdef TINYLM_PARALLEL
    tinylm_core_parallel u_core (
`else
    tinylm_core u_core (
`endif
        .clk(clk), .rst_n(rst_n), .push(push), .token_in(push_tok),
        .start(start), .noise_en(noise_en), .busy(busy), .done(done), .token_out(token_out));

    wire [4:0] rx_tok;
    wire [7:0] echo_ch, gen_ch;
    tinylm_char2tok u_c2t  (.ch(rx_data),   .tok(rx_tok));
    tinylm_tok2char u_echo (.tok(rx_tok),   .ch(echo_ch));
    tinylm_tok2char u_gen  (.tok(token_out), .ch(gen_ch));

    localparam [3:0] S_WAIT = 0, S_SEND = 1, S_SENDING = 2, S_CRLF1 = 3, S_GEN = 4,
                     S_GEN_WAIT = 5, S_GEN_NEXT = 6, S_END = 7, S_END_LF = 8;
    reg [3:0] state, after_send;
    reg       gen_after_crlf;   // CR LF before or after the generated text?
    reg [$clog2(GEN_LEN+1)-1:0] remaining;

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_WAIT; after_send <= S_WAIT; tx_start <= 0; tx_data <= 0;
            push <= 0; start <= 0; push_tok <= 0; noise_en <= 0; remaining <= 0;
            gen_after_crlf <= 0;
        end else begin
            tx_start <= 1'b0;
            push     <= 1'b0;
            start    <= 1'b0;
            case (state)
            S_WAIT: if (rx_valid) begin
                if (rx_data == 8'h0D || rx_data == 8'h0A) begin          // Enter
                    tx_data <= 8'h0D; state <= S_SEND; after_send <= S_CRLF1;
                    gen_after_crlf <= 1'b1;
                end else if (rx_data == "~") begin
                    noise_en <= ~noise_en;
                    tx_data  <= "~"; state <= S_SEND; after_send <= S_WAIT;
                end else begin
                    push <= 1'b1; push_tok <= rx_tok;
                    tx_data <= echo_ch; state <= S_SEND; after_send <= S_WAIT;
                end
            end
            // generic "send tx_data, then go to after_send"
            S_SEND:    begin tx_start <= 1'b1; state <= S_SENDING; end
            S_SENDING: if (tx_done) state <= after_send;
            S_CRLF1:   begin
                tx_data <= 8'h0A; state <= S_SEND;
                after_send <= gen_after_crlf ? S_GEN : S_WAIT;
                remaining  <= GEN_LEN;
            end
            S_GEN:      begin start <= 1'b1; state <= S_GEN_WAIT; end
            S_GEN_WAIT: if (done) begin
                tx_data <= gen_ch; state <= S_SEND; after_send <= S_GEN_NEXT;
            end
            S_GEN_NEXT: begin
                remaining <= remaining - 1'b1;
                state     <= (remaining == 1) ? S_END : S_GEN;
            end
            S_END: begin
                gen_after_crlf <= 1'b0;
                tx_data <= 8'h0D; state <= S_SEND; after_send <= S_CRLF1;
            end
            default: state <= S_WAIT;
            endcase
        end
    end
endmodule
