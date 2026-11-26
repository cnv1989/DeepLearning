// Two-flop synchroniser + rising-edge detector.
// Use it for any signal that comes from outside the chip's clock domain
// (a button, a pin driven by a microcontroller, ...). The first flop may go
// metastable; the second gives it a full clock period to settle.
// `rise` is a single-cycle pulse, 3 clocks after the input goes high.
module sync_edge (
    input  wire clk,
    input  wire rst_n,
    input  wire in,
    output wire level,   // synchronised level
    output wire rise     // one-cycle pulse on 0 -> 1
);
    reg [2:0] s;
    always @(posedge clk) begin
        if (!rst_n) s <= 3'b000;
        else        s <= {s[1:0], in};
    end
    assign level = s[1];
    assign rise  = s[1] & ~s[2];
endmodule
