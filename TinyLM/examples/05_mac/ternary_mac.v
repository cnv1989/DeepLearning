// Lesson 3: a ternary multiply-accumulate unit -- the heart of TinyLM.
// With weights restricted to {-1, 0, +1} there is no multiplier at all:
// the "product" is a 3-way mux, so the MAC is just one adder.
module ternary_mac #(
    parameter integer X_W   = 4,   // signed input width
    parameter integer ACC_W = 16
) (
    input  wire                    clk,
    input  wire                    clear,   // load acc with bias instead of accumulating
    input  wire                    en,
    input  wire signed [ACC_W-1:0] bias,
    input  wire        [1:0]       w,       // 00 = 0, 01 = +1, 11 = -1
    input  wire signed [X_W-1:0]   x,
    output reg  signed [ACC_W-1:0] acc
);
    wire signed [ACC_W-1:0] xe   = x;                       // sign-extend
    wire signed [ACC_W-1:0] prod = (w == 2'b01) ?  xe :
                                   (w == 2'b11) ? -xe : {ACC_W{1'b0}};
    always @(posedge clk)
        if (en) acc <= (clear ? bias : acc) + prod;
endmodule
