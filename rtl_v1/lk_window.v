// =============================================================================
// lk_window.v  --  Lucas-Kanade optical flow, Stage 2: 5x5 window accumulator
//
// Input: the gradient stream (Ix, Iy, It) from Stage 1, raster order, one set per pixel.
// Output: for each pixel, the sums of the 5 products over its 5x5 window:
//        Sxx = ΣIx²   Sxy = ΣIx·Iy   Syy = ΣIy²   Sxt = ΣIx·It   Syt = ΣIy·It
//       positions within 2 px of the border output 0. Bit-exact with window_sums in python/lk_golden.py.
//
// How (the same line-buffer pattern, but storing gradients, 4 rows):
//   1. 4 line buffers hold the last 4 rows of gradients. When the gradient at row r, column c arrives,
//      it is combined with rows r-1..r-4 of the same column, 5 rows -> this column's 5 vertical sums (column sums).
//   2. A 5-deep shift register adds the last 5 column sums horizontally -> the 5x5 window sum.
//   3. The window completed by input (r, c) is centred at (r-2, c-2), so output lags input by 2 rows + 2 pixels.
//      To keep the output in strict raster order with exactly one output per input:
//        for c >= 2 the output centre is (r-2, c-2)
//        for c <  2 the output centre is (r-3, W-2+c)  -- the last two pixels of the previous row, always border, value 0
//      The last 2W+2 outputs (all border zeros) are emitted by the FLUSH state at the end of a frame.
//
// Widths: gradient 9-bit signed -> product 18 bits -> column sum 20 bits -> window sum 22 bits (24-bit output for headroom)
// =============================================================================
`timescale 1ns/1ps

module lk_window #(
    parameter W  = 32,
    parameter H  = 24,
    parameter CW = 16,
    parameter SW = 24                   // output width
)(
    input  wire                 clk,
    input  wire                 rst,

    input  wire                 in_valid,
    output wire                 in_ready,
    input  wire signed [8:0]    in_ix,
    input  wire signed [8:0]    in_iy,
    input  wire signed [8:0]    in_it,

    output reg                  out_valid,
    output reg  [CW-1:0]        out_x,
    output reg  [CW-1:0]        out_y,
    output reg  signed [SW-1:0] out_sxx,
    output reg  signed [SW-1:0] out_sxy,
    output reg  signed [SW-1:0] out_syy,
    output reg  signed [SW-1:0] out_sxt,
    output reg  signed [SW-1:0] out_syt,
    output reg                  frame_done
);

    // -------------------------------------------------------------------------
    // line buffers: each entry holds {Ix, Iy, It}, 27 bits
    // -------------------------------------------------------------------------
    reg [26:0] lb1 [0:W-1];     // row r-1
    reg [26:0] lb2 [0:W-1];     // row r-2
    reg [26:0] lb3 [0:W-1];     // row r-3
    reg [26:0] lb4 [0:W-1];     // row r-4

    localparam S_STREAM = 1'b0;
    localparam S_FLUSH  = 1'b1;

    reg             state;
    reg [CW-1:0]    col, row;           // coordinates of the current input

    // stage-1 pipeline registers (column sums) and their coordinate tags
    reg                     s1_v;
    reg [CW-1:0]            s1_col, s1_row;
    reg signed [SW-1:0]     cs_xx, cs_xy, cs_yy, cs_xt, cs_yt;

    // horizontal shift register: the last 4 column sums (h1 = 1 column back ... h4 = 4 columns back)
    reg signed [SW-1:0]     h1_xx, h2_xx, h3_xx, h4_xx;
    reg signed [SW-1:0]     h1_xy, h2_xy, h3_xy, h4_xy;
    reg signed [SW-1:0]     h1_yy, h2_yy, h3_yy, h4_yy;
    reg signed [SW-1:0]     h1_xt, h2_xt, h3_xt, h4_xt;
    reg signed [SW-1:0]     h1_yt, h2_yt, h3_yt, h4_yt;

    // output coordinates used by FLUSH
    reg [CW-1:0]    fx, fy;

    // the last input has entered stage 1 -> FLUSH next cycle, so no new data is accepted this cycle
    wire last_in_flight = s1_v && (s1_row == H-1) && (s1_col == W-1);

    assign in_ready = (state == S_STREAM) && !last_in_flight && !rst;
    wire   fire     = in_valid & in_ready;

    // -------------------------------------------------------------------------
    // combinational: fetch the gradients of 5 rows, form 25 products, add vertically into 5 column sums
    // -------------------------------------------------------------------------
    wire [26:0] g0 = {in_ix, in_iy, in_it};     // row r (the current input)
    wire [26:0] g1 = lb1[col];
    wire [26:0] g2 = lb2[col];
    wire [26:0] g3 = lb3[col];
    wire [26:0] g4 = lb4[col];

    // unpack: each 27-bit word = {Ix[26:18], Iy[17:9], It[8:0]}
    wire signed [8:0] x0 = g0[26:18], y0 = g0[17:9], t0 = g0[8:0];
    wire signed [8:0] x1 = g1[26:18], y1 = g1[17:9], t1 = g1[8:0];
    wire signed [8:0] x2 = g2[26:18], y2 = g2[17:9], t2 = g2[8:0];
    wire signed [8:0] x3 = g3[26:18], y3 = g3[17:9], t3 = g3[8:0];
    wire signed [8:0] x4 = g4[26:18], y4 = g4[17:9], t4 = g4[8:0];

    // 9-bit x 9-bit signed multiply with the result declared as 18 bits (|product| <= 65025 < 2^17).
    // Note: written directly inside a 24-bit add expression, Verilog would first extend the operands to 24 bits
    // and synthesis would use a 24x24 multiplier (2 DSPs), wasting twice the DSPs.
    wire signed [17:0] m_xx0 = x0*x0, m_xx1 = x1*x1, m_xx2 = x2*x2, m_xx3 = x3*x3, m_xx4 = x4*x4;
    wire signed [17:0] m_xy0 = x0*y0, m_xy1 = x1*y1, m_xy2 = x2*y2, m_xy3 = x3*y3, m_xy4 = x4*y4;
    wire signed [17:0] m_yy0 = y0*y0, m_yy1 = y1*y1, m_yy2 = y2*y2, m_yy3 = y3*y3, m_yy4 = y4*y4;
    wire signed [17:0] m_xt0 = x0*t0, m_xt1 = x1*t1, m_xt2 = x2*t2, m_xt3 = x3*t3, m_xt4 = x4*t4;
    wire signed [17:0] m_yt0 = y0*t0, m_yt1 = y1*t1, m_yt2 = y2*t2, m_yt3 = y3*t3, m_yt4 = y4*t4;

    // add the 5 18-bit products -> at most 20 bits, sign-extended into SW bits
    wire signed [SW-1:0] v_xx = m_xx0 + m_xx1 + m_xx2 + m_xx3 + m_xx4;
    wire signed [SW-1:0] v_xy = m_xy0 + m_xy1 + m_xy2 + m_xy3 + m_xy4;
    wire signed [SW-1:0] v_yy = m_yy0 + m_yy1 + m_yy2 + m_yy3 + m_yy4;
    wire signed [SW-1:0] v_xt = m_xt0 + m_xt1 + m_xt2 + m_xt3 + m_xt4;
    wire signed [SW-1:0] v_yt = m_yt0 + m_yt1 + m_yt2 + m_yt3 + m_yt4;

    // -------------------------------------------------------------------------
    // line-buffer writes (no reset, so RAM can be inferred)
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (fire) begin
            lb4[col] <= g3;
            lb3[col] <= g2;
            lb2[col] <= g1;
            lb1[col] <= g0;
        end
    end

    // -------------------------------------------------------------------------
    // stage 1: input counting + registered column sums
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            col <= 0; row <= 0; s1_v <= 1'b0;
            s1_col <= 0; s1_row <= 0;
            cs_xx <= 0; cs_xy <= 0; cs_yy <= 0; cs_xt <= 0; cs_yt <= 0;
        end else begin
            s1_v <= fire;
            if (fire) begin
                s1_col <= col;  s1_row <= row;
                cs_xx  <= v_xx; cs_xy  <= v_xy; cs_yy <= v_yy;
                cs_xt  <= v_xt; cs_yt  <= v_yt;
                if (col == W-1) begin
                    col <= 0;
                    row <= (row == H-1) ? {CW{1'b0}} : row + 1'b1;
                end else begin
                    col <= col + 1'b1;
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // stage 2: horizontal sum + output (with the FLUSH state machine)
    // -------------------------------------------------------------------------
    // output centre coordinates for the current event
    wire          wrap  = (s1_col < 2);
    wire [CW-1:0] ctr_x = wrap ? (W - 2 + s1_col) : (s1_col - 2);
    wire [CW-1:0] ctr_y = wrap ? (s1_row - 3)     : (s1_row - 2);
    wire          emit  = wrap ? (s1_row >= 3)    : (s1_row >= 2);
    // inner: centre (r-2, c-2) is off the border, i.e. 2 <= r-2 <= H-3 and 2 <= c-2 <= W-3,
    //        i.e. r >= 4 and c >= 4 (r <= H-1 and c <= W-1 always hold)
    wire          inner = !wrap && (s1_row >= 4) && (s1_col >= 4);

    always @(posedge clk) begin
        if (rst) begin
            state <= S_STREAM;
            out_valid <= 1'b0; frame_done <= 1'b0;
            out_x <= 0; out_y <= 0;
            out_sxx <= 0; out_sxy <= 0; out_syy <= 0; out_sxt <= 0; out_syt <= 0;
            fx <= 0; fy <= 0;
            h1_xx <= 0; h2_xx <= 0; h3_xx <= 0; h4_xx <= 0;
            h1_xy <= 0; h2_xy <= 0; h3_xy <= 0; h4_xy <= 0;
            h1_yy <= 0; h2_yy <= 0; h3_yy <= 0; h4_yy <= 0;
            h1_xt <= 0; h2_xt <= 0; h3_xt <= 0; h4_xt <= 0;
            h1_yt <= 0; h2_yt <= 0; h3_yt <= 0; h4_yt <= 0;
        end else begin
            out_valid  <= 1'b0;
            frame_done <= 1'b0;

            // horizontal shift register: shift by one for every column sum
            if (s1_v) begin
                h4_xx <= h3_xx; h3_xx <= h2_xx; h2_xx <= h1_xx; h1_xx <= cs_xx;
                h4_xy <= h3_xy; h3_xy <= h2_xy; h2_xy <= h1_xy; h1_xy <= cs_xy;
                h4_yy <= h3_yy; h3_yy <= h2_yy; h2_yy <= h1_yy; h1_yy <= cs_yy;
                h4_xt <= h3_xt; h3_xt <= h2_xt; h2_xt <= h1_xt; h1_xt <= cs_xt;
                h4_yt <= h3_yt; h3_yt <= h2_yt; h2_yt <= h1_yt; h1_yt <= cs_yt;
            end

            case (state)
            S_STREAM: begin
                if (s1_v && emit) begin
                    out_valid <= 1'b1;
                    out_x <= ctr_x;
                    out_y <= ctr_y;
                    out_sxx <= inner ? (cs_xx + h1_xx + h2_xx + h3_xx + h4_xx) : {SW{1'b0}};
                    out_sxy <= inner ? (cs_xy + h1_xy + h2_xy + h3_xy + h4_xy) : {SW{1'b0}};
                    out_syy <= inner ? (cs_yy + h1_yy + h2_yy + h3_yy + h4_yy) : {SW{1'b0}};
                    out_sxt <= inner ? (cs_xt + h1_xt + h2_xt + h3_xt + h4_xt) : {SW{1'b0}};
                    out_syt <= inner ? (cs_yt + h1_yt + h2_yt + h3_yt + h4_yt) : {SW{1'b0}};
                end
                if (last_in_flight) begin
                    // the last input of a frame has output centre (H-3, W-3); the rest is emitted from (H-3, W-2)
                    state <= S_FLUSH;
                    fx <= W - 2;
                    fy <= H - 3;
                end
            end
            S_FLUSH: begin
                out_valid <= 1'b1;
                out_x <= fx;  out_y <= fy;
                out_sxx <= 0; out_sxy <= 0; out_syy <= 0; out_sxt <= 0; out_syt <= 0;
                if (fx == W-1) begin
                    fx <= 0;
                    if (fy == H-1) begin
                        state      <= S_STREAM;
                        frame_done <= 1'b1;
                    end else begin
                        fy <= fy + 1'b1;
                    end
                end else begin
                    fx <= fx + 1'b1;
                end
            end
            endcase
        end
    end

endmodule
