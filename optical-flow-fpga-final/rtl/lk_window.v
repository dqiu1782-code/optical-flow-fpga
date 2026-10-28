// =============================================================================
// lk_window.v  --  Lucas-Kanade optical flow, Stage 2: 5x5 window accumulator (v2, 250 MHz pipeline)
//
// Input: the gradient stream (Ix, Iy, It) from Stage 1, raster order.
// Output: for each pixel, the sums of the 5 products over its 5x5 window:
//        Sxx = ΣIx²   Sxy = ΣIx·Iy   Syy = ΣIy²   Sxt = ΣIx·It   Syt = ΣIy·It
//       positions within 2 px of the border output 0. Bit-exact with window_sums in python/lk_golden.py.
//
// How:
//   1. A line buffer keeps the last 4 rows of gradients. When row r, column c arrives, together with rows r-1..r-4
//      of the same column it gives this column's 5 "column sums" (sums of the 5 products vertically).
//   2. Adding the last 5 column sums horizontally gives the 5x5 window sum.
//   3. The window completed by input (r, c) is centred at (r-2, c-2):
//        c >= 2: output centre (r-2, c-2);  c < 2: output centre (r-3, W-2+c) (end of the previous row, border, value 0)
//      The last 2W+2 outputs of a frame (all border zeros) are emitted by the FLUSH state.
//
// v2: run-time size cfg_w/cfg_h; what used to be one cycle of "read buffer -> multiply -> add 5 -> add 5 again"
//     is split into a 6-stage pipeline with one multiply or one adder level per stage, targeting 250 MHz (4 ns per cycle):
//       A  read the line buffer (synchronous read -> Block RAM)
//       B  25 multiplies (9x9 -> 18 bits)
//       C  column sum, first level: m0+m1, m2+m3, m4
//       D  column sum, second level: add three -> column sum cs
//       E  horizontal, first level: cs+h1, h2+h3, h4          (FLUSH zeros are inserted here)
//       F  horizontal, second level: add three -> output registers
// =============================================================================
`timescale 1ns/1ps

module lk_window #(
    parameter WMAX = 1024,
    parameter CW   = 16,
    parameter SW   = 24                 // output width
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire [CW-1:0]        cfg_w,
    input  wire [CW-1:0]        cfg_h,

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

    localparam AW = (WMAX > 1) ? $clog2(WMAX) : 1;

    // line buffer: one wide RAM, each address holds {Ix,Iy,It} of 4 rows = 4 x 27 = 108 bits
    //   lb[c] = {g(r-1,c), g(r-2,c), g(r-3,c), g(r-4,c)}
    (* ram_style = "block" *) reg [107:0] lb [0:WMAX-1];

    localparam S_STREAM = 1'b0;
    localparam S_FLUSH  = 1'b1;

    reg             state;
    reg             tail;               // last pixel of the frame received; no new data until FLUSH ends
    reg [CW-1:0]    col, row;
    reg [CW-1:0]    wm1, hm1;

    // register the size once (also updated during reset, so it is right as soon as reset ends)
    always @(posedge clk) begin
        wm1 <= cfg_w - 1'b1;
        hm1 <= cfg_h - 1'b1;
    end

    assign in_ready = (state == S_STREAM) && !tail && !rst;
    wire   fire     = in_valid & in_ready;

    // ---------------- A: read the line buffer ----------------
    reg             a_v;
    reg [CW-1:0]    a_col, a_row;
    reg [26:0]      a_g0;
    reg [107:0]     a_old;              // Block RAM output register

    always @(posedge clk) begin
        if (fire) a_old <= lb[col[AW-1:0]];
        // write back one cycle later (address a_col); the next read of this address is a row later, long after
        if (a_v)  lb[a_col[AW-1:0]] <= {a_g0, a_old[107:27]};
    end

    always @(posedge clk) begin
        if (rst) begin
            a_v <= 1'b0; a_col <= 0; a_row <= 0; a_g0 <= 0;
            col <= 0; row <= 0; tail <= 1'b0;
        end else begin
            a_v <= fire;
            if (fire) begin
                a_col <= col; a_row <= row;
                a_g0  <= {in_ix, in_iy, in_it};
                if (col == wm1) begin
                    col <= 0;
                    if (row == hm1) begin
                        row  <= 0;
                        tail <= 1'b1;
                    end else
                        row <= row + 1'b1;
                end else
                    col <= col + 1'b1;
            end
            if (frame_done) tail <= 1'b0;
        end
    end

    // ---------------- B: 25 multiplies ----------------
    wire [26:0] g0 = a_g0;
    wire [26:0] g1 = a_old[107:81];
    wire [26:0] g2 = a_old[80:54];
    wire [26:0] g3 = a_old[53:27];
    wire [26:0] g4 = a_old[26:0];
    wire signed [8:0] x0 = g0[26:18], y0 = g0[17:9], t0 = g0[8:0];
    wire signed [8:0] x1 = g1[26:18], y1 = g1[17:9], t1 = g1[8:0];
    wire signed [8:0] x2 = g2[26:18], y2 = g2[17:9], t2 = g2[8:0];
    wire signed [8:0] x3 = g3[26:18], y3 = g3[17:9], t3 = g3[8:0];
    wire signed [8:0] x4 = g4[26:18], y4 = g4[17:9], t4 = g4[8:0];

    reg             b_v;
    reg [CW-1:0]    b_col, b_row;
    reg signed [17:0] m_xx0, m_xx1, m_xx2, m_xx3, m_xx4;
    reg signed [17:0] m_xy0, m_xy1, m_xy2, m_xy3, m_xy4;
    reg signed [17:0] m_yy0, m_yy1, m_yy2, m_yy3, m_yy4;
    reg signed [17:0] m_xt0, m_xt1, m_xt2, m_xt3, m_xt4;
    reg signed [17:0] m_yt0, m_yt1, m_yt2, m_yt3, m_yt4;

    always @(posedge clk) begin
        b_v <= rst ? 1'b0 : a_v;
        b_col <= a_col;  b_row <= a_row;
        // results are explicitly 18 bits (|product| <= 65025), so they synthesise to 9x9 multipliers
        m_xx0 <= x0*x0; m_xx1 <= x1*x1; m_xx2 <= x2*x2; m_xx3 <= x3*x3; m_xx4 <= x4*x4;
        m_xy0 <= x0*y0; m_xy1 <= x1*y1; m_xy2 <= x2*y2; m_xy3 <= x3*y3; m_xy4 <= x4*y4;
        m_yy0 <= y0*y0; m_yy1 <= y1*y1; m_yy2 <= y2*y2; m_yy3 <= y3*y3; m_yy4 <= y4*y4;
        m_xt0 <= x0*t0; m_xt1 <= x1*t1; m_xt2 <= x2*t2; m_xt3 <= x3*t3; m_xt4 <= x4*t4;
        m_yt0 <= y0*t0; m_yt1 <= y1*t1; m_yt2 <= y2*t2; m_yt3 <= y3*t3; m_yt4 <= y4*t4;
    end

    // ---------------- C: column sum, first level ----------------
    reg             c_v;
    reg [CW-1:0]    c_col, c_row;
    reg signed [18:0] c_xx01, c_xx23, c_xy01, c_xy23, c_yy01, c_yy23, c_xt01, c_xt23, c_yt01, c_yt23;
    reg signed [17:0] c_xx4, c_xy4, c_yy4, c_xt4, c_yt4;

    always @(posedge clk) begin
        c_v <= rst ? 1'b0 : b_v;
        c_col <= b_col;  c_row <= b_row;
        c_xx01 <= m_xx0 + m_xx1;  c_xx23 <= m_xx2 + m_xx3;  c_xx4 <= m_xx4;
        c_xy01 <= m_xy0 + m_xy1;  c_xy23 <= m_xy2 + m_xy3;  c_xy4 <= m_xy4;
        c_yy01 <= m_yy0 + m_yy1;  c_yy23 <= m_yy2 + m_yy3;  c_yy4 <= m_yy4;
        c_xt01 <= m_xt0 + m_xt1;  c_xt23 <= m_xt2 + m_xt3;  c_xt4 <= m_xt4;
        c_yt01 <= m_yt0 + m_yt1;  c_yt23 <= m_yt2 + m_yt3;  c_yt4 <= m_yt4;
    end

    // ---------------- D: column sum, second level -> cs (with coordinate tags s1_col/s1_row) ----------------
    reg                     s1_v;
    reg [CW-1:0]            s1_col, s1_row;
    reg signed [SW-1:0]     cs_xx, cs_xy, cs_yy, cs_xt, cs_yt;

    // output coordinates and the "is border" decision are also precomputed at this stage (registers); stage E only selects
    reg             s1_wrap, s1_emit, s1_inner, s1_last;
    reg [CW-1:0]    s1_cx, s1_cy;
    wire            c_wrap = (c_col < 2);

    always @(posedge clk) begin
        s1_v <= rst ? 1'b0 : c_v;
        s1_col <= c_col;  s1_row <= c_row;
        s1_wrap  <= c_wrap;
        s1_cx    <= c_wrap ? (wm1 - 1'b1 + c_col) : (c_col - 2'd2);
        s1_cy    <= c_wrap ? (c_row - 2'd3)       : (c_row - 2'd2);
        s1_emit  <= c_wrap ? (c_row >= 3)         : (c_row >= 2);
        s1_inner <= !c_wrap && (c_row >= 4) && (c_col >= 4);
        s1_last  <= (c_row == hm1) && (c_col == wm1);
        cs_xx <= c_xx01 + c_xx23 + c_xx4;
        cs_xy <= c_xy01 + c_xy23 + c_xy4;
        cs_yy <= c_yy01 + c_yy23 + c_yy4;
        cs_xt <= c_xt01 + c_xt23 + c_xt4;
        cs_yt <= c_yt01 + c_yt23 + c_yt4;
    end

    // ---------------- E: horizontal first level (+ FLUSH state machine) ----------------
    reg signed [SW-1:0]     h1_xx, h2_xx, h3_xx, h4_xx;
    reg signed [SW-1:0]     h1_xy, h2_xy, h3_xy, h4_xy;
    reg signed [SW-1:0]     h1_yy, h2_yy, h3_yy, h4_yy;
    reg signed [SW-1:0]     h1_xt, h2_xt, h3_xt, h4_xt;
    reg signed [SW-1:0]     h1_yt, h2_yt, h3_yt, h4_yt;

    reg                     e_v, e_zero, e_last;
    reg [CW-1:0]            e_x, e_y;
    reg signed [SW-1:0]     e_xx_a, e_xx_b, e_xx_c, e_xy_a, e_xy_b, e_xy_c, e_yy_a, e_yy_b, e_yy_c;
    reg signed [SW-1:0]     e_xt_a, e_xt_b, e_xt_c, e_yt_a, e_yt_b, e_yt_c;
    reg [CW-1:0]            fx, fy;

    wire [CW-1:0] ctr_x   = s1_cx;
    wire [CW-1:0] ctr_y   = s1_cy;
    wire          emit    = s1_emit;
    wire          inner   = s1_inner;
    wire          last_ev = s1_v && s1_last;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_STREAM;
            e_v <= 1'b0; e_zero <= 1'b1; e_last <= 1'b0; e_x <= 0; e_y <= 0;
            fx <= 0; fy <= 0;
            h1_xx <= 0; h2_xx <= 0; h3_xx <= 0; h4_xx <= 0;
            h1_xy <= 0; h2_xy <= 0; h3_xy <= 0; h4_xy <= 0;
            h1_yy <= 0; h2_yy <= 0; h3_yy <= 0; h4_yy <= 0;
            h1_xt <= 0; h2_xt <= 0; h3_xt <= 0; h4_xt <= 0;
            h1_yt <= 0; h2_yt <= 0; h3_yt <= 0; h4_yt <= 0;
        end else begin
            e_v    <= 1'b0;
            e_last <= 1'b0;
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
                    e_v <= 1'b1;  e_zero <= !inner;
                    e_x <= ctr_x; e_y <= ctr_y;
                end
                if (last_ev) begin
                    state <= S_FLUSH;           // the rest is emitted starting from (H-3, W-2)
                    fx <= wm1 - 1'b1;
                    fy <= hm1 - 2'd2;
                end
            end
            S_FLUSH: begin
                e_v <= 1'b1;  e_zero <= 1'b1;
                e_x <= fx;    e_y <= fy;
                if (fx == wm1) begin
                    fx <= 0;
                    if (fy == hm1) begin
                        state  <= S_STREAM;
                        e_last <= 1'b1;
                    end else
                        fy <= fy + 1'b1;
                end else
                    fx <= fx + 1'b1;
            end
            endcase
        end
    end

    // horizontal partial sums (datapath not reset, saves resources)
    always @(posedge clk) begin
        e_xx_a <= cs_xx + h1_xx;  e_xx_b <= h2_xx + h3_xx;  e_xx_c <= h4_xx;
        e_xy_a <= cs_xy + h1_xy;  e_xy_b <= h2_xy + h3_xy;  e_xy_c <= h4_xy;
        e_yy_a <= cs_yy + h1_yy;  e_yy_b <= h2_yy + h3_yy;  e_yy_c <= h4_yy;
        e_xt_a <= cs_xt + h1_xt;  e_xt_b <= h2_xt + h3_xt;  e_xt_c <= h4_xt;
        e_yt_a <= cs_yt + h1_yt;  e_yt_b <= h2_yt + h3_yt;  e_yt_c <= h4_yt;
    end

    // ---------------- F: horizontal second level -> output ----------------
    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0; frame_done <= 1'b0;
            out_x <= 0; out_y <= 0;
            out_sxx <= 0; out_sxy <= 0; out_syy <= 0; out_sxt <= 0; out_syt <= 0;
        end else begin
            out_valid  <= e_v;
            frame_done <= e_last;
            out_x <= e_x;  out_y <= e_y;
            out_sxx <= e_zero ? {SW{1'b0}} : (e_xx_a + e_xx_b + e_xx_c);
            out_sxy <= e_zero ? {SW{1'b0}} : (e_xy_a + e_xy_b + e_xy_c);
            out_syy <= e_zero ? {SW{1'b0}} : (e_yy_a + e_yy_b + e_yy_c);
            out_sxt <= e_zero ? {SW{1'b0}} : (e_xt_a + e_xt_b + e_xt_c);
            out_syt <= e_zero ? {SW{1'b0}} : (e_yt_a + e_yt_b + e_yt_c);
        end
    end

endmodule
