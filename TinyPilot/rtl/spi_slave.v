// SPI slave, mode 0 (CPOL=0, CPHA=0), MSB first, oversampled by the system clock.
// SCK must be at most clk/8.
//
// Each frame (CS low ... CS high):
//   MOSI: 13 bytes = sp_roll, sp_pitch, sp_yaw, gyro_roll, gyro_pitch, gyro_yaw (int16, big endian), throttle
//   MISO: 4 bytes  = motors from the PREVIOUS inference (m0..m3), then zeros
// A frame of exactly 104 bits pulses `frame_valid`; any other length (e.g. a
// 4-byte read) just reads the motors without starting an inference.
module spi_slave (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         sck,
    input  wire         mosi,
    input  wire         cs_n,
    output wire         miso,
    input  wire [31:0]  tx_word,      // sampled when CS falls
    output reg  [103:0] frame,
    output reg          frame_valid
);
    reg [2:0] sck_s, cs_s;
    reg [1:0] mosi_s;
    always @(posedge clk) begin
        if (!rst_n) begin sck_s <= 0; cs_s <= 3'b111; mosi_s <= 0; end
        else begin
            sck_s  <= {sck_s[1:0], sck};
            cs_s   <= {cs_s[1:0], cs_n};
            mosi_s <= {mosi_s[0], mosi};
        end
    end
    wire sck_rise = sck_s[1] & ~sck_s[2];
    wire sck_fall = ~sck_s[1] & sck_s[2];
    wire cs_fall  = ~cs_s[1] & cs_s[2];
    wire cs_rise  = cs_s[1] & ~cs_s[2];
    wire active   = ~cs_s[1];

    reg [6:0]  nbits;
    reg [31:0] tx;
    assign miso = tx[31];

    always @(posedge clk) begin
        if (!rst_n) begin
            nbits <= 0; tx <= 0; frame <= 0; frame_valid <= 0;
        end else begin
            frame_valid <= 1'b0;
            if (cs_fall) begin
                nbits <= 0;
                tx    <= tx_word;
            end else if (active) begin
                if (sck_rise) begin
                    frame <= {frame[102:0], mosi_s[1]};
                    if (nbits != 7'd127) nbits <= nbits + 1'b1;
                end
                if (sck_fall) tx <= {tx[30:0], 1'b0};
            end
            if (cs_rise && nbits == 7'd104) frame_valid <= 1'b1;
        end
    end
endmodule
