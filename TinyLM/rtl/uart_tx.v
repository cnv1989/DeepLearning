// UART transmitter, 8N1. Pulse `start` for one cycle while !busy;
// `done` pulses for one cycle after the stop bit.
module uart_tx #(
    parameter integer CLKS_PER_BIT = 104    // 12 MHz / 115200 baud
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       start,
    input  wire [7:0] data,
    output reg        tx,
    output wire       busy,
    output reg        done
);
    reg [9:0]  shreg;                 // {stop, data[7:0], start}
    reg [3:0]  bits_left;
    reg [$clog2(CLKS_PER_BIT)-1:0] cnt;

    assign busy = (bits_left != 0);

    always @(posedge clk) begin
        if (!rst_n) begin
            tx <= 1'b1; shreg <= 10'h3FF; bits_left <= 0; cnt <= 0; done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (!busy) begin
                if (start) begin
                    shreg     <= {1'b1, data, 1'b0};
                    bits_left <= 4'd10;
                    cnt       <= 0;
                    tx        <= 1'b0;          // start bit goes out immediately
                end
            end else if (cnt == CLKS_PER_BIT - 1) begin
                cnt       <= 0;
                shreg     <= {1'b1, shreg[9:1]};
                bits_left <= bits_left - 1'b1;
                tx        <= shreg[1];
                if (bits_left == 1) begin
                    tx   <= 1'b1;
                    done <= 1'b1;
                end
            end else begin
                cnt <= cnt + 1'b1;
            end
        end
    end
endmodule
