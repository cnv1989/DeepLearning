// TinyPilot core: integer PID-style front end + hard-wired neural network.
// Bit-exact twin of model/pilot.py (FrontEnd + IntNet).
//
//   start (1 cycle) with sp/gyro/thr valid ->  1 cycle  front end (3 axes in parallel)
//                                           -> H*N_IN  layer-1 MACs (one per clock)
//                                           -> N_OUT*H layer-2 MACs
//                                           -> done (1 cycle), motors updated
// Latency: 1 + 24*10 + 4*24 = 337 cycles (6.7 us at 50 MHz); the control loop is 1 ms.
module pilot_core (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [47:0] sp,       // {roll, pitch, yaw} signed int16 deg/s
    input  wire [47:0] gyro,     // {roll, pitch, yaw} signed int16 deg/s
    input  wire [7:0]  thr,      // 0..255
    output reg         busy,
    output reg         done,
    output reg  [31:0] motors    // {m0, m1, m2, m3}, 0..255 each
);
`include "pilot_params.vh"

    localparam [1:0] S_IDLE = 0, S_L1 = 1, S_L2 = 2;
    reg [1:0] state;

    // ------------------------------------------------------------------
    // Front end: per-axis state and feature computation (combinational)
    // ------------------------------------------------------------------
    reg  signed [15:0] prev [0:2];
    reg  signed [15:0] df4  [0:2];
    reg  signed [19:0] iacc [0:2];
    reg  signed [7:0]  f    [0:N_IN-1];    // feature memory
    reg  [7:0]         h    [0:H-1];       // hidden activations

    wire signed [15:0] sp_a   [0:2];
    wire signed [15:0] g_a    [0:2];
    wire signed [7:0]  fe_n   [0:2];
    wire signed [7:0]  fd_n   [0:2];
    wire signed [7:0]  fi_n   [0:2];
    wire signed [15:0] df4_n  [0:2];
    wire signed [19:0] iacc_n [0:2];
    wire               ireset = (thr < THR_IRESET);

    function signed [7:0] sat8(input signed [19:0] x);
        sat8 = (x > 127) ? 8'sd127 : (x < -127) ? -8'sd127 : x[7:0];
    endfunction

    genvar a;
    generate for (a = 0; a < 3; a = a + 1) begin : axis
        localparam integer LIM = (a == 0) ? I_LIM_R : (a == 1) ? I_LIM_P : I_LIM_Y;
        assign sp_a[a] = sp[47 - 16*a -: 16];
        assign g_a[a]  = gyro[47 - 16*a -: 16];
        wire signed [19:0] e     = sp_a[a] - g_a[a];
        wire signed [19:0] isum  = iacc[a] + e;
        assign iacc_n[a] = ireset       ? 20'sd0 :
                           (isum >  LIM) ? LIM :
                           (isum < -LIM) ? -LIM : isum;
        wire signed [19:0] d     = prev[a] - g_a[a];
        wire signed [19:0] dsum  = df4[a] + d - (df4[a] >>> 2);
        assign df4_n[a] = (dsum > 32767) ? 16'sh7FFF : (dsum < -32768) ? 16'sh8000 : dsum[15:0];
        assign fe_n[a]  = sat8(e >>> 1);
        assign fd_n[a]  = sat8(df4_n[a] >>> 1);
        assign fi_n[a]  = sat8(iacc_n[a] >>> 8);
    end endgenerate

    // ------------------------------------------------------------------
    // Network datapath: one signed 8x9 multiply-accumulate per clock
    // ------------------------------------------------------------------
    reg  [IN_W-1:0]  i;
    reg  [J_W-1:0]   j;
    reg  [K_W-1:0]   k;
    reg  [W1_AW-1:0] w1_addr;
    reg  [W2_AW-1:0] w2_addr;
    reg  signed [ACC_W-1:0] acc;

    wire signed [W_BITS-1:0] w1, w2;
    wire signed [B_BITS-1:0] b1, b2;
    pilot_w1_rom u_w1 (.addr(w1_addr), .data(w1));
    pilot_w2_rom u_w2 (.addr(w2_addr), .data(w2));
    pilot_b1_rom u_b1 (.addr(j),       .data(b1));
    pilot_b2_rom u_b2 (.addr(k),       .data(b2));

    wire               in_l1 = (state == S_L1);
    wire signed [8:0]  x     = in_l1 ? {f[i][7], f[i]} : {1'b0, h[j]};
    wire signed [W_BITS-1:0] w = in_l1 ? w1 : w2;
    wire signed [ACC_W-1:0] prod = w * x;                  // sign-extends to ACC_W
    wire first = in_l1 ? (i == 0) : (j == 0);
    wire signed [ACC_W-1:0] bias     = in_l1 ? b1 : b2;
    wire signed [ACC_W-1:0] acc_next = (first ? bias : acc) + prod;

    wire signed [ACC_W-1:0] r1 = (acc_next + (1 <<< (S1 - 1))) >>> S1;
    wire signed [ACC_W-1:0] r2 = (acc_next + (1 <<< (S2 - 1))) >>> S2;
    wire [7:0] h_new = (acc_next <= 0) ? 8'd0 : (r1 > 255) ? 8'd255 : r1[7:0];
    wire [7:0] m_new = (r2 < 0) ? 8'd0 : (r2 > 255) ? 8'd255 : r2[7:0];

    // ------------------------------------------------------------------
    // Control
    // ------------------------------------------------------------------
    integer n;
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; busy <= 0; done <= 0; motors <= 0; acc <= 0;
            i <= 0; j <= 0; k <= 0; w1_addr <= 0; w2_addr <= 0;
            for (n = 0; n < 3; n = n + 1) begin prev[n] <= 0; df4[n] <= 0; iacc[n] <= 0; end
            for (n = 0; n < N_IN; n = n + 1) f[n] <= 0;
            for (n = 0; n < H; n = n + 1) h[n] <= 0;
        end else begin
            done <= 1'b0;
            case (state)
            S_IDLE: if (start) begin
                for (n = 0; n < 3; n = n + 1) begin
                    prev[n] <= g_a[n];
                    df4[n]  <= df4_n[n];
                    iacc[n] <= iacc_n[n];
                    f[n]     <= fe_n[n];
                    f[3 + n] <= fd_n[n];
                    f[6 + n] <= fi_n[n];
                end
                f[9] <= {1'b0, thr[7:1]};
                busy <= 1'b1;
                state <= S_L1;
                i <= 0; j <= 0; k <= 0; w1_addr <= 0; w2_addr <= 0;
            end

            S_L1: begin
                acc <= acc_next;
                w1_addr <= w1_addr + 1'b1;
                if (i == N_IN - 1) begin
                    i <= 0;
                    h[j] <= h_new;
                    if (j == H - 1) begin j <= 0; state <= S_L2; end
                    else j <= j + 1'b1;
                end else i <= i + 1'b1;
            end

            S_L2: begin
                acc <= acc_next;
                w2_addr <= w2_addr + 1'b1;
                if (j == H - 1) begin
                    j <= 0;
                    motors[31 - 8*k -: 8] <= m_new;
                    if (k == N_OUT - 1) begin
                        k <= 0; busy <= 1'b0; done <= 1'b1; state <= S_IDLE;
                    end else k <= k + 1'b1;
                end else j <= j + 1'b1;
            end

            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
