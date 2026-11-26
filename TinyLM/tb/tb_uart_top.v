// End-to-end test of the FPGA design: types into the serial port like a
// terminal would and checks every byte that comes back.
`timescale 1ns/1ps
module tb_uart_top;
    localparam integer CPB = 16;          // fast baud rate for simulation
    localparam integer GEN_LEN = 24;      // must match GEN_LEN_UART in model/gen_vectors.py

    reg clk = 0;
    always #10 clk = ~clk;
    reg rst_n = 0, rx = 1;
    wire tx, noise_en;

    tinylm_uart_top #(.CLKS_PER_BIT(CPB), .GEN_LEN(GEN_LEN)) dut (
        .clk(clk), .rst_n(rst_n), .rx(rx), .tx(tx), .noise_en(noise_en));

    // Decode what the design sends using our own uart_rx.
    wire [7:0] got;
    wire       got_valid;
    uart_rx #(.CLKS_PER_BIT(CPB)) mon (.clk(clk), .rst_n(rst_n), .rx(tx), .data(got), .valid(got_valid));

`include "uart_counts.vh"
    reg [7:0] send [0:UART_IN_LEN];
    reg [7:0] exp  [0:UART_EXP_LEN-1];
    integer n_got = 0, errors = 0, i, b;

    always @(posedge clk) if (got_valid) begin
        if (got !== exp[n_got]) begin
            errors = errors + 1;
            $display("\nMISMATCH byte %0d: got %h expected %h", n_got, got, exp[n_got]);
        end
        if (got == 8'h0D) ; else $write("%c", got);
        n_got = n_got + 1;
    end

    task uart_send(input [7:0] c);
        begin
            rx = 0; repeat (CPB) @(posedge clk);                 // start bit
            for (b = 0; b < 8; b = b + 1) begin
                rx = c[b]; repeat (CPB) @(posedge clk);
            end
            rx = 1; repeat (CPB) @(posedge clk);                 // stop bit
        end
    endtask

    initial begin
        $readmemh("tb/vectors/uart_in.hex", send);
        $readmemh("tb/vectors/uart_expected.hex", exp);
        repeat (5) @(posedge clk);
        rst_n = 1;
        for (i = 0; i < UART_IN_LEN; i = i + 1) begin
            uart_send(send[i]);
            // A human types slowly: wait until the design is idle again.
            repeat (4 * CPB) @(posedge clk);
            wait (dut.state == 0 && !dut.u_tx.busy);
        end
        repeat (20 * CPB) @(posedge clk);
        $display("\ntb_uart_top: %0d/%0d bytes received, %0d errors", n_got, UART_EXP_LEN, errors);
        if (errors == 0 && n_got == UART_EXP_LEN) $display("PASS");
        else                                       $display("FAIL");
        $finish;
    end

    initial begin
        #200_000_000;
        $display("FAIL: timeout");
        $finish;
    end
endmodule
