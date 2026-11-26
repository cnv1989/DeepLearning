// TinyLM sequential core -- one multiply-accumulate (MAC) per clock.
//
// Network (all integer, see model/ref_model.py for the bit-exact spec):
//   x   = concat(E[ctx[0]], ..., E[ctx[N-1]])            N*D signed E_BITS values
//   h_j = clamp(round(relu(b1_j + sum_i W1[j][i]*x_i) / 2^S1), 0, 2^H_BITS-1)
//   z_k = b2_k + sum_j W2[k][j]*h_j  (+ LFSR noise when noise_en)
//   token_out = argmax_k z_k   (first maximum wins)
//
// Weights are ternary (-1/0/+1), so the "multiplier" is just a mux that picks
// +x, -x or 0. The weights live in hard-wired ROMs (rtl/generated/tinylm_weights.v).
//
// Protocol:
//   push  (1 cycle, while !busy): shift token_in into the context window
//   start (1 cycle, while !busy): run one inference; when finished `done`
//          pulses for one cycle, token_out holds the prediction, and the
//          prediction is automatically shifted into the context (so you can
//          just keep pulsing start to generate text).
// Latency: H*N*D + V*H cycles (tt config: 16*12 + 32*16 = 704 cycles).
module tinylm_core (
    input  wire       clk,
    input  wire       rst_n,      // synchronous, active low
    input  wire       push,
    input  wire [4:0] token_in,
    input  wire       start,
    input  wire       noise_en,
    output reg        busy,
    output reg        done,
    output reg  [4:0] token_out
);
`include "tinylm_params.vh"

    localparam [1:0] S_IDLE = 2'd0, S_L1 = 2'd1, S_L2 = 2'd2;
    localparam [H_BITS-1:0] HMAX = {H_BITS{1'b1}};

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------
    reg  [1:0]          state;
    reg  [N*5-1:0]      ctx;        // ctx[p*5 +: 5], p = 0 oldest .. N-1 newest
    reg  [P_W-1:0]      p;          // layer 1: which context position
    reg  [LOG2D-1:0]    d;          // layer 1: which embedding element
    reg  [J_W-1:0]      j;          // hidden neuron (layer 1) / hidden input (layer 2)
    reg  [K_W-1:0]      k;          // output logit (layer 2)
    reg  [W1_AW-1:0]    w1_addr;
    reg  [W2_AW-1:0]    w2_addr;
    reg  signed [ACC_W-1:0] acc;
    reg  signed [ACC_W-1:0] best_val;
    reg  [4:0]          best_idx;
    reg  [15:0]         lfsr;
    reg  [H_BITS-1:0]   h_mem [0:H-1];

    // ------------------------------------------------------------------
    // Hard-wired ROMs
    // ------------------------------------------------------------------
    wire [4:0]                tok = ctx[p*5 +: 5];
    wire signed [E_BITS-1:0]  e_val;
    wire [1:0]                w1, w2;
    wire signed [B_BITS-1:0]  b1, b2;

    tinylm_emb_rom u_emb (.addr({tok, d}), .data(e_val));
    tinylm_w1_rom  u_w1  (.addr(w1_addr),  .data(w1));
    tinylm_b1_rom  u_b1  (.addr(j),        .data(b1));
    tinylm_w2_rom  u_w2  (.addr(w2_addr),  .data(w2));
    tinylm_b2_rom  u_b2  (.addr(k[K_W-1:0]), .data(b2));

    // ------------------------------------------------------------------
    // Datapath: one ternary MAC
    // ------------------------------------------------------------------
    // Layer 1 operand: sign-extended embedding value. Layer 2 operand: zero-extended h.
    wire signed [ACC_W-1:0] x1 = e_val;                              // sign-extends
    wire signed [ACC_W-1:0] x2 = $signed({{(ACC_W-H_BITS){1'b0}}, h_mem[j]});
    wire        in_l1   = (state == S_L1);
    wire [1:0]  w       = in_l1 ? w1 : w2;
    wire signed [ACC_W-1:0] x    = in_l1 ? x1 : x2;
    wire signed [ACC_W-1:0] prod = (w == 2'b01) ?  x :
                                   (w == 2'b11) ? -x : {ACC_W{1'b0}};

    wire first1 = (p == 0) && (d == 0);
    wire last1  = (p == N - 1) && (d == D - 1);
    wire first2 = (j == 0);
    wire last2  = (j == H - 1);
    wire first  = in_l1 ? first1 : first2;
    wire signed [ACC_W-1:0] bias     = in_l1 ? b1 : b2;   // sign-extends
    wire signed [ACC_W-1:0] acc_base = first ? bias : acc;
    wire signed [ACC_W-1:0] acc_next = acc_base + prod;

    // Requantise layer-1 result: relu, round-half-up shift, saturate.
    wire signed [ACC_W-1:0] shifted = (acc_next + (1 <<< (S1 - 1))) >>> S1;
    wire [H_BITS-1:0] h_new = (acc_next <= 0)    ? {H_BITS{1'b0}} :
                              (shifted > HMAX)   ? HMAX : shifted[H_BITS-1:0];

    // Layer-2 result: optional noise, then running argmax.
    wire signed [ACC_W-1:0] noise = noise_en ? $signed({{(ACC_W-NOISE_BITS){1'b0}}, lfsr[NOISE_BITS-1:0]})
                                             : {ACC_W{1'b0}};
    wire signed [ACC_W-1:0] logit = acc_next + noise;
    wire        take     = (k == 0) || (logit > best_val);
    wire [4:0]  final_idx = take ? k : best_idx;
    wire [15:0] lfsr_next = {1'b0, lfsr[15:1]} ^ (lfsr[0] ? 16'hB400 : 16'h0000);

    // ------------------------------------------------------------------
    // Control FSM
    // ------------------------------------------------------------------
    integer n;
    always @(posedge clk) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            ctx       <= {N*5{1'b0}};
            busy      <= 1'b0;
            done      <= 1'b0;
            token_out <= 5'd0;
            lfsr      <= 16'hACE1;
            acc       <= 0;
            best_val  <= 0;
            best_idx  <= 0;
            p <= 0; d <= 0; j <= 0; k <= 0; w1_addr <= 0; w2_addr <= 0;
            for (n = 0; n < H; n = n + 1) h_mem[n] <= 0;
        end else begin
            done <= 1'b0;
            case (state)
            S_IDLE: begin
                if (start) begin
                    state <= S_L1;
                    busy  <= 1'b1;
                    p <= 0; d <= 0; j <= 0; k <= 0; w1_addr <= 0; w2_addr <= 0;
                end else if (push) begin
                    ctx <= {token_in, ctx[N*5-1:5]};
                end
            end

            S_L1: begin                      // H * N * D cycles
                acc     <= acc_next;
                w1_addr <= w1_addr + 1'b1;
                if (d == D - 1) begin
                    d <= 0;
                    p <= (p == N - 1) ? {P_W{1'b0}} : p + 1'b1;
                end else begin
                    d <= d + 1'b1;
                end
                if (last1) begin
                    h_mem[j] <= h_new;
                    if (j == H - 1) begin
                        j     <= 0;
                        state <= S_L2;
                    end else begin
                        j <= j + 1'b1;
                    end
                end
            end

            S_L2: begin                      // V * H cycles
                acc     <= acc_next;
                w2_addr <= w2_addr + 1'b1;
                j       <= last2 ? {J_W{1'b0}} : j + 1'b1;
                if (last2) begin
                    lfsr <= lfsr_next;
                    if (take) begin
                        best_val <= logit;
                        best_idx <= k;
                    end
                    if (k == V - 1) begin
                        token_out <= final_idx;
                        ctx       <= {final_idx, ctx[N*5-1:5]};
                        done      <= 1'b1;
                        busy      <= 1'b0;
                        state     <= S_IDLE;
                    end else begin
                        k <= k + 1'b1;
                    end
                end
            end

            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
