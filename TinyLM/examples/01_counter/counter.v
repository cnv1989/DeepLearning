// Lesson 1: the "hello world" of hardware -- a counter with enable and
// synchronous reset. Compare with the VHDL you remember:
//   process(clk) begin if rising_edge(clk) then if rst='1' then q<=(others=>'0');
//   elsif en='1' then q<=q+1; end if; end if; end process;
module counter #(
    parameter integer WIDTH = 4
) (
    input  wire             clk,
    input  wire             rst,     // synchronous, active high
    input  wire             en,
    output reg  [WIDTH-1:0] q,
    output wire             wrap     // combinational: high when q is at its max
);
    always @(posedge clk) begin      // sequential logic -> non-blocking (<=)
        if (rst)     q <= {WIDTH{1'b0}};
        else if (en) q <= q + 1'b1;
    end

    assign wrap = &q;                // reduction AND: all bits 1
endmodule
