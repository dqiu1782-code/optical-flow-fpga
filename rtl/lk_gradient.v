// =============================================================================
// lk_gradient.v  --  Lucas-Kanade optical flow, Stage 1: streaming gradient unit (v2)
//
// Function: two frames I0 (frame t) and I1 (frame t+1) stream in together in raster order,
//       and each pixel outputs three gradients:
//         Ix(y,x) = I0(y, x+1) - I0(y, x-1)
//         Iy(y,x) = I0(y+1, x) - I0(y-1, x)
//         It(y,x) = I1(y, x)   - I0(y, x)
//       the outermost ring of the image (border) outputs 0. Bit-exact with python/lk_golden.py.
//
// Line-buffer idea: when row r, column c arrives, the data for centre pixel (r-1, c) is complete:
//            col:   c-1        c         c+1
//   row r-2:               lb0_r2[c]
//   row r-1:   left        up          upr          <- centre row (line buffer lb0_r1)
//   row r  :               in_p0
//
// v2 changes (for 250 MHz + the pyramid):
//   1. Image size is a run-time input cfg_w / cfg_h (each pyramid level has its own size and shares this hardware);
//      the line buffers are sized for the maximum width WMAX.
//   2. Line buffers use synchronous reads (address this cycle, data next cycle), so synthesis can put them
//      in Block RAM instead of distributed RAM built from many LUTs plus a 1024:1 multiplexer
//      (that multiplexer was one of the slowest paths at 250 MHz).
//      Cost: output is 1 cycle later. lb0_r1 needs c (up) and c+1 (upr) at once, more than one Block RAM read port,
//      so it prefetches: column c+1 is read while column c is processed, and next cycle it is exactly the new up.
// =============================================================================
`timescale 1ns/1ps

module lk_gradient #(
    parameter WMAX = 1024,          // maximum supported image width (sets the line-buffer size)
    parameter CW   = 16             // coordinate counter width
)(
    input  wire                 clk,
    input  wire                 rst,        // synchronous reset, active high
    input  wire [CW-1:0]        cfg_w,      // image width (constant within a frame)
    input  wire [CW-1:0]        cfg_h,      // image height

    input  wire                 in_valid,
    output wire                 in_ready,
    input  wire [7:0]           in_p0,      // I0(r, c)
    input  wire [7:0]           in_p1,      // I1(r, c)

    output reg                  out_valid,
    output reg  [CW-1:0]        out_x,
    output reg  [CW-1:0]        out_y,
    output reg  signed [8:0]    out_ix,
    output reg  signed [8:0]    out_iy,
    output reg  signed [8:0]    out_it,
    output reg                  frame_done
);

    localparam AW = (WMAX > 1) ? $clog2(WMAX) : 1;

    // line buffers (synchronous read, inferable as Block RAM)
    (* ram_style = "block" *) reg [7:0] lb0_r1 [0:WMAX-1];    // I0 row r-1
    (* ram_style = "block" *) reg [7:0] lb0_r2 [0:WMAX-1];    // I0 row r-2
    (* ram_style = "block" *) reg [7:0] lb1_r1 [0:WMAX-1];    // I1 row r-1

    localparam S_STREAM = 1'b0;
    localparam S_FLUSH  = 1'b1;

    reg             state;
    reg [CW-1:0]    col, row;
    reg [CW-1:0]    wm1, hm1;       // cfg_w-1, cfg_h-1 (registered once, off the critical path)
    reg [7:0]       prev_up;        // = I0(r-1, c-1)

    // register the size once (also updated during reset, so it is right as soon as reset ends)
    always @(posedge clk) begin
        wm1 <= cfg_w - 1'b1;
        hm1 <= cfg_h - 1'b1;
    end

    assign in_ready = (state == S_STREAM) && !rst;
    wire   fire     = in_valid & in_ready;

    wire [AW-1:0] a_col = col[AW-1:0];
    wire [AW-1:0] a_nxt = (col == wm1) ? {AW{1'b0}} : (col[AW-1:0] + 1'b1);

    // ---------------- stage G0: read line buffers (synchronous) + register the current pixel ----------------
    reg             g0_v, g0_flush;
    reg [CW-1:0]    g0_col, g0_row;
    reg [7:0]       g0_p0, g0_up, g0_left;
    reg [7:0]       q_up2, q_upi1, q_upr;   // Block RAM output registers
    // q_upr = the previously prefetched lb0_r1[c+1]; for the next pixel it is exactly up = I0(r-1, c)
    wire [7:0]      nxt_up = q_upr;

    always @(posedge clk) begin
        if (fire) begin
            // read-first: a read of the same address returns the old value before the write
            q_up2  <= lb0_r2[a_col];
            q_upi1 <= lb1_r1[a_col];
            q_upr  <= lb0_r1[a_nxt];        // prefetch the next column
            lb0_r2[a_col] <= nxt_up;        // the previous row "sinks" into the row before it
            lb0_r1[a_col] <= in_p0;
            lb1_r1[a_col] <= in_p1;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= S_STREAM;
            col <= 0; row <= 0;
            prev_up <= 8'd0;
            g0_v <= 1'b0; g0_flush <= 1'b0;
            g0_col <= 0; g0_row <= 0;
            g0_p0 <= 0; g0_up <= 0; g0_left <= 0;
        end else begin
            g0_v     <= 1'b0;
            g0_flush <= 1'b0;
            case (state)
            S_STREAM: if (fire) begin
                g0_v    <= 1'b1;
                g0_col  <= col;     g0_row <= row;
                g0_p0   <= in_p0;
                g0_up   <= nxt_up;
                g0_left <= prev_up;
                prev_up <= nxt_up;
                if (col == wm1) begin
                    col <= 0;
                    if (row == hm1) begin
                        row   <= 0;
                        state <= S_FLUSH;
                    end else
                        row <= row + 1'b1;
                end else
                    col <= col + 1'b1;
            end
            S_FLUSH: begin
                // emit the last row (border, all 0) through stage G0 too, so the order stays intact
                g0_v     <= 1'b1;
                g0_flush <= 1'b1;
                g0_col   <= col;
                if (col == wm1) begin
                    col   <= 0;
                    state <= S_STREAM;
                end else
                    col <= col + 1'b1;
            end
            endcase
        end
    end

    // ---------------- stage G1: subtraction + output registers ----------------
    wire signed [8:0] ix_raw = $signed({1'b0, q_upr})  - $signed({1'b0, g0_left});
    wire signed [8:0] iy_raw = $signed({1'b0, g0_p0})  - $signed({1'b0, q_up2});
    wire signed [8:0] it_raw = $signed({1'b0, q_upi1}) - $signed({1'b0, g0_up});
    wire border = g0_flush || (g0_row == 1) || (g0_col == 0) || (g0_col == wm1);

    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0; frame_done <= 1'b0;
            out_x <= 0; out_y <= 0;
            out_ix <= 0; out_iy <= 0; out_it <= 0;
        end else begin
            // row 0 only fills the buffers and outputs nothing; FLUSH entries output row H-1
            out_valid  <= g0_v && (g0_flush || g0_row != 0);
            frame_done <= g0_v && g0_flush && (g0_col == wm1);
            out_x  <= g0_col;
            out_y  <= g0_flush ? hm1 : (g0_row - 1'b1);
            out_ix <= border ? 9'sd0 : ix_raw;
            out_iy <= border ? 9'sd0 : iy_raw;
            out_it <= border ? 9'sd0 : it_raw;
        end
    end

endmodule
