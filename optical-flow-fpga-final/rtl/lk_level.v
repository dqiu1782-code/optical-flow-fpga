// =============================================================================
// lk_level.v  --  per-level engine of pyramid LK + per-frame statistics unit
//
//   coarser-level flow --> [lk_upflow x2 upsample] --+
//   I0 at this level --------------------------------+--> [lk_warp] --> [lk_top 3-stage LK] --> du,dv
//   I1 at this level -----------------------------------+                    |
//                                   g (guess) --> [FIFO] ---------------> [+] --> u = g + du --> output stream / stats
//
// The same hardware processes the levels from coarsest to finest (sizes are run-time configuration).
//   - Intermediate results (u, v) go back to DDR (small: level 1 has 1/4 of the pixels) and are read back as the next guess;
//   - The finest level (full resolution) only feeds the statistics unit; the dense flow map need not be written (cfg_dense=1 writes it too).
// In this project the testbench plays the DMA + scheduler (on the board this is a small driver on the ARM core).
// =============================================================================
`timescale 1ns/1ps

// -----------------------------------------------------------------------------
// lk_upflow: upsample the coarser flow (Wc x Hc) to this level (W x H) and pair it with the I0 pixels
//   g(y,x) = 2 * flow_c(min(y>>1, Hc-1), min(x>>1, Wc-1))
//   Even rows read one row of coarse flow from the input stream (and store it in a line buffer); odd rows (and the last row
//   when H is odd) replay the line buffer -- so the coarse flow is read from DDR only once.
// -----------------------------------------------------------------------------
module lk_upflow #(
    parameter WMAX = 1024,
    parameter CW   = 16
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire [CW-1:0]        cfg_w,          // size of this level; coarser level = this level >> 1
    input  wire [CW-1:0]        cfg_h,
    input  wire                 cfg_guess,      // 0 = coarsest level, guess is 0, no coarse flow is read

    input  wire                 p_valid,        // I0 pixels
    output wire                 p_ready,
    input  wire [7:0]           p_p0,

    input  wire                 c_valid,        // coarser-level flow (raster order)
    output wire                 c_ready,
    input  wire signed [15:0]   c_u,
    input  wire signed [15:0]   c_v,

    output wire                 a_valid,
    input  wire                 a_ready,
    output wire [7:0]           a_p0,
    output wire signed [16:0]   a_gu2,          // 2 x coarse flow (not yet clamped)
    output wire signed [16:0]   a_gv2
);
    localparam HB = (WMAX > 2) ? $clog2(WMAX / 2) : 1;

    (* ram_style = "distributed" *) reg [31:0] rb [0:WMAX/2-1];   // one row of coarse flow {u, v}
    reg [31:0]   hold;                  // coarse-flow sample just read
    reg [CW-1:0] x, y, wm1, hm1, wc, hc;

    always @(posedge clk) begin
        wm1 <= cfg_w - 1'b1;
        hm1 <= cfg_h - 1'b1;
        wc  <= cfg_w >> 1;
        hc  <= cfg_h >> 1;
    end

    // All per-pixel decisions are precomputed one cycle early in registers (250 MHz):
    //   fresh_r  this row reads new coarse flow from the stream;  need_r  this pixel consumes a new sample
    //   cxc_r    min(x>>1, Wc-1);          eol_r   x == W-1
    reg           fresh_r, need_r, eol_r;
    reg [CW-1:0]  cx_r, cxc_r;
    wire          need_c = need_r;
    wire          fresh  = fresh_r;
    // The output side is a 2-entry FIFO ("skid buffer"): accepting data depends only on whether it is full (a register),
    // not on the downstream a_ready -- so ready never ripples combinationally through several modules (the enemy of 250 MHz)
    (* ram_style = "distributed" *) reg [41:0] sk [0:1];
    reg         sk_wp, sk_rp;
    reg  [1:0]  sk_cnt;
    wire        out_en = (sk_cnt != 2'd2);
    wire        a_pop  = a_valid && a_ready;

    assign p_ready = !rst && out_en && (!need_c || c_valid);
    assign c_ready = !rst && out_en && need_c && p_valid;
    wire   fire    = p_valid && p_ready;

    wire [31:0] cur = need_c ? {c_u, c_v} : (fresh ? hold : rb[cxc_r[HB-1:0]]);
    wire signed [15:0] gu = cfg_guess ? cur[31:16] : 16'sd0;
    wire signed [15:0] gv = cfg_guess ? cur[15:0]  : 16'sd0;

    always @(posedge clk) begin
        if (fire && need_c) rb[cx_r[HB-1:0]] <= {c_u, c_v};
        if (fire) sk[sk_wp] <= {p_p0, gu, 1'b0, gv, 1'b0};
    end

    wire [41:0] sk_head = sk[sk_rp];
    assign a_valid = (sk_cnt != 2'd0);
    assign a_p0    = sk_head[41:34];
    assign a_gu2   = sk_head[33:17];
    assign a_gv2   = sk_head[16:0];

    // coordinates of the next pixel -> precomputed decisions
    wire [CW-1:0] x_n  = fire ? (eol_r ? {CW{1'b0}} : x + 1'b1) : x;
    wire [CW-1:0] y_n  = (fire && eol_r) ? ((y == hm1) ? {CW{1'b0}} : y + 1'b1) : y;
    wire [CW-1:0] cx_n = x_n >> 1;
    wire          fresh_n = cfg_guess && !y_n[0] && ((y_n >> 1) < hc);

    always @(posedge clk) begin
        if (rst) begin
            x <= 0; y <= 0; hold <= 0;
            sk_wp <= 1'b0; sk_rp <= 1'b0; sk_cnt <= 2'd0;
            fresh_r <= 1'b0; need_r <= 1'b0; eol_r <= 1'b0; cx_r <= 0; cxc_r <= 0;
        end else begin
            if (fire)  sk_wp <= ~sk_wp;
            if (a_pop) sk_rp <= ~sk_rp;
            sk_cnt <= sk_cnt + fire - a_pop;
            if (fire && need_c) hold <= {c_u, c_v};
            x <= x_n;
            y <= y_n;
            fresh_r <= fresh_n;
            need_r  <= fresh_n && !x_n[0] && (cx_n < wc);
            eol_r   <= (x_n == wm1);
            cx_r    <= cx_n;
            cxc_r   <= (cx_n < wc) ? cx_n : (wc - 1'b1);
        end
    end
endmodule


// -----------------------------------------------------------------------------
// lk_stats: per-frame statistics unit (only pixels with flag=1 count)
//   count, sum u, sum v, sum(|u|+|v|), max(|u|+|v|), 8-bin histogram (|u|+|v| binned at 0.5/1/2/4/8/16/32 px)
//   Keyframe selection only needs these numbers, so the dense flow map does not have to go back to memory.
// -----------------------------------------------------------------------------
module lk_stats #(
    parameter NW = 24,              // count width (up to 16 M pixels)
    parameter SWD = 40              // sum width
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 clear,          // clear before each frame
    input  wire                 in_valid,
    input  wire signed [15:0]   in_u,
    input  wire signed [15:0]   in_v,
    input  wire                 in_flag,

    output reg  [NW-1:0]        st_count,
    output reg  signed [SWD-1:0] st_sum_u,
    output reg  signed [SWD-1:0] st_sum_v,
    output reg  [SWD-1:0]       st_sum_l1,
    output reg  [16:0]          st_max_l1,
    output reg  [8*NW-1:0]      st_hist         // bin k at [NW*k +: NW]
);
    // T1: absolute values
    reg               t1_v;
    reg signed [15:0] t1_u, t1_v_;
    reg [15:0]        t1_au, t1_av;
    // T2:|u|+|v|
    reg               t2_v;
    reg signed [15:0] t2_u, t2_vv;
    reg [16:0]        t2_l1;
    // T3: binning
    reg               t3_v;
    reg signed [15:0] t3_u, t3_vv;
    reg [16:0]        t3_l1;
    reg [2:0]         t3_bin;

    wire [2:0] bin = (t2_l1 < 17'd128)  ? 3'd0 : (t2_l1 < 17'd256)  ? 3'd1 :
                     (t2_l1 < 17'd512)  ? 3'd2 : (t2_l1 < 17'd1024) ? 3'd3 :
                     (t2_l1 < 17'd2048) ? 3'd4 : (t2_l1 < 17'd4096) ? 3'd5 :
                     (t2_l1 < 17'd8192) ? 3'd6 : 3'd7;

    integer k;
    always @(posedge clk) begin
        if (rst) begin
            t1_v <= 0; t2_v <= 0; t3_v <= 0;
        end else begin
            t1_v <= in_valid && in_flag && !clear;
            t2_v <= t1_v && !clear;
            t3_v <= t2_v && !clear;
        end
        t1_u <= in_u;   t1_v_ <= in_v;
        t1_au <= in_u[15] ? -in_u : in_u;   // |-32767| = 32767, no overflow (outputs are clamped to +-32767)
        t1_av <= in_v[15] ? -in_v : in_v;
        t2_u <= t1_u;   t2_vv <= t1_v_;
        t2_l1 <= {1'b0, t1_au} + {1'b0, t1_av};
        t3_u <= t2_u;   t3_vv <= t2_vv;   t3_l1 <= t2_l1;   t3_bin <= bin;

        if (rst || clear) begin
            st_count <= 0; st_sum_u <= 0; st_sum_v <= 0; st_sum_l1 <= 0; st_max_l1 <= 0;
            st_hist <= 0;
        end else if (t3_v) begin
            st_count  <= st_count + 1'b1;
            st_sum_u  <= st_sum_u + t3_u;
            st_sum_v  <= st_sum_v + t3_vv;
            st_sum_l1 <= st_sum_l1 + t3_l1;
            if (t3_l1 > st_max_l1) st_max_l1 <= t3_l1;
            for (k = 0; k < 8; k = k + 1)
                if (t3_bin == k) st_hist[NW*k +: NW] <= st_hist[NW*k +: NW] + 1'b1;
        end
    end
endmodule


// -----------------------------------------------------------------------------
// lk_gfifo: synchronous FIFO (stores each pixel's guess g until lk_top produces du to add back)
//   Reads are synchronous (address on the pop cycle, dout valid the next cycle), so it maps to Block RAM.
// -----------------------------------------------------------------------------
module lk_gfifo #(
    parameter DW = 32,
    parameter AB = 12
)(
    input  wire          clk,
    input  wire          rst,
    input  wire          push,
    input  wire [DW-1:0] din,
    input  wire          pop,
    output reg  [DW-1:0] dout,          // valid one cycle after pop
    output reg  [AB:0]   count,
    output reg  [AB:0]   max_count      // simulation/debug: peak occupancy
);
    (* ram_style = "block" *) reg [DW-1:0] mem [0:(1<<AB)-1];
    reg [AB-1:0] wp, rp;
    always @(posedge clk) begin
        if (push) mem[wp] <= din;
        if (pop)  dout <= mem[rp];
        if (rst) begin
            wp <= 0; rp <= 0; count <= 0; max_count <= 0;
        end else begin
            if (push) wp <= wp + 1'b1;
            if (pop)  rp <= rp + 1'b1;
            count <= count + push - pop;
            if (count > max_count) max_count <= count;
        end
    end
endmodule


// -----------------------------------------------------------------------------
// lk_level: per-level engine of one lane
// -----------------------------------------------------------------------------
module lk_level #(
    parameter WMAX = 1024,
    parameter NRB  = 6,
    parameter CW   = 16,
    parameter [63:0] DET_MIN = 64'd65536
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire [CW-1:0]        cfg_w,          // size of this level
    input  wire [CW-1:0]        cfg_h,
    input  wire                 cfg_guess,      // whether there is a coarser-level guess
    input  wire                 cfg_stats,      // feed this level's result to the statistics unit
    input  wire [14:0]          cfg_dumax,      // per-level residual du/dv clamp (Q8), 0 = off
    input  wire                 stats_clear,

    input  wire                 p_valid,        // I0 at this level
    output wire                 p_ready,
    input  wire [7:0]           p_p0,
    input  wire                 b_valid,        // I1 at this level
    output wire                 b_ready,
    input  wire [7:0]           b_p1,
    input  wire                 c_valid,        // coarser-level flow
    output wire                 c_ready,
    input  wire signed [15:0]   c_u,
    input  wire signed [15:0]   c_v,

    output reg                  f_valid,        // flow at this level, u = g + du (Q8)
    output reg  [CW-1:0]        f_x,
    output reg  [CW-1:0]        f_y,
    output reg  signed [15:0]   f_u,
    output reg  signed [15:0]   f_v,
    output reg                  f_flag,

    output wire [23:0]          st_count,
    output wire signed [39:0]   st_sum_u,
    output wire signed [39:0]   st_sum_v,
    output wire [39:0]          st_sum_l1,
    output wire [16:0]          st_max_l1,
    output wire [8*24-1:0]      st_hist
);
    // ---------------- upsample the coarse flow ----------------
    wire                a_valid, a_ready;
    wire [7:0]          a_p0;
    wire signed [16:0]  a_gu2, a_gv2;
    lk_upflow #(.WMAX(WMAX), .CW(CW)) u_up (
        .clk(clk), .rst(rst), .cfg_w(cfg_w), .cfg_h(cfg_h), .cfg_guess(cfg_guess),
        .p_valid(p_valid), .p_ready(p_ready), .p_p0(p_p0),
        .c_valid(c_valid), .c_ready(c_ready), .c_u(c_u), .c_v(c_v),
        .a_valid(a_valid), .a_ready(a_ready), .a_p0(a_p0), .a_gu2(a_gu2), .a_gv2(a_gv2));

    // ---------------- warp ----------------
    wire                w_valid, w_ready;
    wire [7:0]          w_p0, w_p1w;
    wire signed [15:0]  w_gu, w_gv;
    lk_warp #(.WMAX(WMAX), .NRB(NRB), .CW(CW)) u_warp (
        .clk(clk), .rst(rst), .cfg_w(cfg_w), .cfg_h(cfg_h),
        .b_valid(b_valid), .b_ready(b_ready), .b_p1(b_p1),
        .a_valid(a_valid), .a_ready(a_ready), .a_p0(a_p0), .a_gu2(a_gu2), .a_gv2(a_gv2),
        .o_valid(w_valid), .o_ready(w_ready), .o_p0(w_p0), .o_p1w(w_p1w), .o_gu(w_gu), .o_gv(w_gv));

    // ---------------- the original 3-stage LK ----------------
    wire                l_valid, l_flag;
    wire [CW-1:0]       l_x, l_y;
    wire signed [15:0]  l_u, l_v;
    lk_top #(.WMAX(WMAX), .CW(CW), .DET_MIN(DET_MIN)) u_lk (
        .clk(clk), .rst(rst), .cfg_w(cfg_w), .cfg_h(cfg_h),
        .in_valid(w_valid), .in_ready(w_ready), .in_p0(w_p0), .in_p1(w_p1w),
        .out_valid(l_valid), .out_x(l_x), .out_y(l_y), .out_u(l_u), .out_v(l_v), .out_flag(l_flag));

    // ---------------- guess FIFO ----------------
    // at most about 3W + 40 pixels are in flight inside lk_top (gradient 1 row + window 2 rows + pipeline)
    localparam FAB = $clog2(3 * WMAX + 64);
    wire [31:0]  g_out;
    wire [FAB:0] g_count, g_max;
    lk_gfifo #(.DW(32), .AB(FAB)) u_fifo (
        .clk(clk), .rst(rst),
        .push(w_valid && w_ready), .din({w_gu, w_gv}),
        .pop(l_valid), .dout(g_out), .count(g_count), .max_count(g_max));

    // ---------------- merge u = clip(g + du) ----------------
    // the FIFO output is one cycle late, so the lk_top output is registered once to align
    reg                 d_valid, d_flag;
    reg [CW-1:0]        d_x, d_y;
    reg signed [15:0]   d_u, d_v;
    // (optional) residual clamp: du beyond +-cfg_dumax is truncated so a bad solution is not amplified into the next level
    wire signed [15:0] dmax = {1'b0, cfg_dumax};
    wire               clampon = (cfg_dumax != 15'd0);
    always @(posedge clk) begin
        d_valid <= rst ? 1'b0 : l_valid;
        d_x <= l_x;  d_y <= l_y;  d_flag <= l_flag;
        d_u <= (clampon && l_u >  dmax) ? dmax : (clampon && l_u < -dmax) ? -dmax : l_u;
        d_v <= (clampon && l_v >  dmax) ? dmax : (clampon && l_v < -dmax) ? -dmax : l_v;
    end

    wire signed [16:0] su = $signed(g_out[31:16]) + d_u;
    wire signed [16:0] sv = $signed(g_out[15:0])  + d_v;
    localparam signed [16:0] QM = 17'sd32767;

    always @(posedge clk) begin
        if (rst) f_valid <= 1'b0;
        else     f_valid <= d_valid;
        f_x <= d_x;  f_y <= d_y;  f_flag <= d_flag;
        f_u <= (su > QM) ? 16'sd32767 : (su < -QM) ? -16'sd32767 : su[15:0];
        f_v <= (sv > QM) ? 16'sd32767 : (sv < -QM) ? -16'sd32767 : sv[15:0];
    end

    // ---------------- per-frame statistics ----------------
    lk_stats #(.NW(24), .SWD(40)) u_stats (
        .clk(clk), .rst(rst), .clear(stats_clear),
        .in_valid(f_valid && cfg_stats), .in_u(f_u), .in_v(f_v), .in_flag(f_flag),
        .st_count(st_count), .st_sum_u(st_sum_u), .st_sum_v(st_sum_v), .st_sum_l1(st_sum_l1),
        .st_max_l1(st_max_l1), .st_hist(st_hist));

endmodule
