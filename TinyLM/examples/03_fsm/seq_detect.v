// Lesson 2: a finite-state machine. Detects the bit pattern 1011 on a serial
// input (overlapping matches allowed). Written in the "two-process" style
// (state register + combinational next-state logic) you probably used in VHDL.
module seq_detect (
    input  wire clk,
    input  wire rst,
    input  wire bit_in,
    output wire found          // Moore output: depends only on the state
);
    localparam [2:0] S0 = 0,   // nothing matched
                     S1 = 1,   // "1"
                     S2 = 2,   // "10"
                     S3 = 3,   // "101"
                     S4 = 4;   // "1011"  -> found
    reg [2:0] state, next;

    always @(posedge clk)                // process 1: the state register
        if (rst) state <= S0;
        else     state <= next;

    always @* begin                      // process 2: next-state logic (blocking =)
        next = state;                    // default assignment avoids latches!
        case (state)
            S0: next = bit_in ? S1 : S0;
            S1: next = bit_in ? S1 : S2;
            S2: next = bit_in ? S3 : S0;
            S3: next = bit_in ? S4 : S2;
            S4: next = bit_in ? S1 : S2;
            default: next = S0;
        endcase
    end

    assign found = (state == S4);
endmodule
