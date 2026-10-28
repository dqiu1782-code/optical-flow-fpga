// =============================================================================
// lk_gradient.v  --  Lucas-Kanade optical flow, Stage 1: streaming gradient unit
//
// Function: two frames I0 (frame t) and I1 (frame t+1) stream in together in raster order,
//       and each pixel outputs three gradients:
//         Ix(y,x) = I0(y, x+1) - I0(y, x-1)
//         Iy(y,x) = I0(y+1, x) - I0(y-1, x)
//         It(y,x) = I1(y, x)   - I0(y, x)
//       the outermost ring of the image (border) outputs 0. Bit-exact with python/lk_golden.py.
//
// Core idea (the standard pattern of image-processing hardware):
//   pixels arrive one at a time and the image cannot be accessed randomly. Iy needs the rows above and below,
//   so line buffers keep the last two rows.
//   When the pixel at row r, column c arrives, everything needed for centre pixel (r-1, c) is available:
//
//            col:   c-1        c         c+1
//   row r-2:               lb0_r2[c]                    <- two rows up (line buffer 2)
//   row r-1:   left_r1     lb0_r1[c]   lb0_r1[c+1]       <- previous row (line buffer 1), the centre row
//   row r  :               in_p0                         <- the row being input
//
//   So the output lags the input by one row. The last row (y = H-1) would wait for an input row H,
//   but it is a border row with all zeros, so a small state machine (FLUSH) emits it directly.
//
// Handshake: the input pixel of a cycle is accepted when in_valid & in_ready are both 1.
//       Upstream may pause at any time (in_valid = 0, a "bubble"); the module waits.
// =============================================================================
`timescale 1ns/1ps

module lk_gradient #(
    parameter W  = 32,              // image width (pixels)
    parameter H  = 24,              // image height (pixels)
    parameter CW = 16               // coordinate counter width (up to 65535 x 65535)
)(
    input  wire                 clk,
    input  wire                 rst,        // synchronous reset, active high

    // ---- input stream: pixels of both frames at the same position ----
    input  wire                 in_valid,
    output wire                 in_ready,
    input  wire [7:0]           in_p0,      // I0(r, c)
    input  wire [7:0]           in_p1,      // I1(r, c)

    // ---- output stream: gradients of centre pixel (out_y, out_x), raster order ----
    output reg                  out_valid,
    output reg  [CW-1:0]        out_x,
    output reg  [CW-1:0]        out_y,
    output reg  signed [8:0]    out_ix,     // 9-bit signed, range -255..255
    output reg  signed [8:0]    out_iy,
    output reg  signed [8:0]    out_it,
    output reg                  frame_done  // high together with the last output of a frame (one cycle only)
);

    // -------------------------------------------------------------------------
    // line buffers: Vivado infers RAM (combinational read -> distributed RAM / LUTRAM)
    // -------------------------------------------------------------------------
    reg [7:0] lb0_r1 [0:W-1];       // I0 row r-1
    reg [7:0] lb0_r2 [0:W-1];       // I0 row r-2
    reg [7:0] lb1_r1 [0:W-1];       // I1 row r-1 (for It)

    // -------------------------------------------------------------------------
    // state machine + coordinate counters
    // -------------------------------------------------------------------------
    localparam S_STREAM = 1'b0;     // normal pixel input
    localparam S_FLUSH  = 1'b1;     // input finished, emitting the last row (all 0)

    reg             state;
    reg [CW-1:0]    col;            // STREAM: column of the current input pixel; FLUSH: column being emitted
    reg [CW-1:0]    row;            // row of the current input pixel
    reg [7:0]       left_r1;        // I0(r-1, c-1)

    assign in_ready = (state == S_STREAM) && !rst;   // no input during reset, so the line buffers stay clean
    wire   fire     = in_valid & in_ready;

    // -------------------------------------------------------------------------
    // combinational: read the window from the line buffers (returns the old value before this cycle's write)
    // -------------------------------------------------------------------------
    wire [7:0] up    = lb0_r1[col];                                 // I0(r-1, c)
    wire [7:0] up2   = lb0_r2[col];                                 // I0(r-2, c)
    wire [7:0] upr   = (col == W-1) ? 8'd0 : lb0_r1[col + 1'b1];    // I0(r-1, c+1); 0 out of range (it is a border anyway)
    wire [7:0] up_i1 = lb1_r1[col];                                 // I1(r-1, c)

    // zero-extend 8-bit unsigned to 9-bit signed, then subtract: result -255..255 fits 9 bits exactly, no overflow
    wire signed [8:0] ix_raw = $signed({1'b0, upr})   - $signed({1'b0, left_r1});
    wire signed [8:0] iy_raw = $signed({1'b0, in_p0}) - $signed({1'b0, up2});
    wire signed [8:0] it_raw = $signed({1'b0, up_i1}) - $signed({1'b0, up});

    // is the centre pixel (row-1, col) on the border?
    // (centre row = H-1 never happens in STREAM; FLUSH handles it)
    wire border = (row == 1) || (col == 0) || (col == W-1);

    // -------------------------------------------------------------------------
    // line-buffer writes (a separate always block without reset, so Vivado infers RAM cleanly)
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (fire) begin
            lb0_r2[col] <= up;      // the previous row "sinks" into the row before it
            lb0_r1[col] <= in_p0;   // the current row becomes the previous row
            lb1_r1[col] <= in_p1;
        end
    end

    // -------------------------------------------------------------------------
    // main sequential logic
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            state      <= S_STREAM;
            col        <= {CW{1'b0}};
            row        <= {CW{1'b0}};
            left_r1    <= 8'd0;
            out_valid  <= 1'b0;
            frame_done <= 1'b0;
            out_x      <= {CW{1'b0}};
            out_y      <= {CW{1'b0}};
            out_ix     <= 9'sd0;
            out_iy     <= 9'sd0;
            out_it     <= 9'sd0;
        end else begin
            out_valid  <= 1'b0;     // no output by default; raised below when the condition holds
            frame_done <= 1'b0;

            case (state)
            // ---------------------------------------------------------------
            S_STREAM: begin
                if (fire) begin
                    left_r1 <= up;  // keep I0(r-1, c); next cycle it is I0(r-1, c-1)
                                    // (next cycle this line-buffer slot is overwritten by the new row, so save it now)

                    // row 0 only fills the buffers; from row 1 on, one centre pixel is output per cycle
                    if (row != 0) begin
                        out_valid <= 1'b1;
                        out_x     <= col;
                        out_y     <= row - 1'b1;
                        out_ix    <= border ? 9'sd0 : ix_raw;
                        out_iy    <= border ? 9'sd0 : iy_raw;
                        out_it    <= border ? 9'sd0 : it_raw;
                    end

                    // advance the coordinates
                    if (col == W-1) begin
                        col <= {CW{1'b0}};
                        if (row == H-1) begin
                            row   <= {CW{1'b0}};
                            state <= S_FLUSH;       // whole frame received
                        end else begin
                            row   <= row + 1'b1;
                        end
                    end else begin
                        col <= col + 1'b1;
                    end
                end
            end
            // ---------------------------------------------------------------
            S_FLUSH: begin
                // emit the last row y = H-1 (border, all 0); no new input needed
                out_valid <= 1'b1;
                out_x     <= col;
                out_y     <= H - 1;
                out_ix    <= 9'sd0;
                out_iy    <= 9'sd0;
                out_it    <= 9'sd0;
                if (col == W-1) begin
                    col        <= {CW{1'b0}};
                    state      <= S_STREAM;     // ready for the next frame
                    frame_done <= 1'b1;
                end else begin
                    col <= col + 1'b1;
                end
            end
            endcase
        end
    end

endmodule
