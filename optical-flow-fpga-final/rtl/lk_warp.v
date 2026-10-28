// =============================================================================
// lk_warp.v  --  warp the second frame back by the initial guess g (bilinear interpolation), streaming
//
// At each pyramid level the second frame I1 is first shifted back by the previous level's result g, giving I1w(y,x) ~= I1(y+gv, x+gu),
// then (I0, I1w) go through the usual LK pipeline, which only has to solve for the small residual.
// Bit-exact with warp_bilinear in python/lk_pyramid.py:
//     gu = clamp(2*uc, ±32767)    gv = clamp(2*vc, ±RY*256)          (Q8)
//     px = clip((x<<8)+gu, 0, ((W-1)<<8)-1)  xi = px>>8  fx = px&255  (same for y)
//     top = I1[yi][xi]*(256-fx) + I1[yi][xi+1]*fx       (= I00*256 + (I01-I00)*fx)
//     bot = I1[yi+1][xi]*(256-fx) + I1[yi+1][xi+1]*fx
//     I1w = (top*(256-fy) + bot*fy + 32768) >> 16
//
// Two input streams:
//   b stream: pixels of I1 at this level (raster order), written into a ring line buffer (NR = 2^NRB rows)
//   a stream: I0 pixels + initial guess (gu2, gv2) = 2 x previous-level flow (not yet clamped)
// Output: (p0, I1w, clamped gu, gv) for lk_top; gu/gv are kept so they can be added back at the end.
//
// Ring line buffer: output row y needs I1 rows y-RY .. y+RY+1 (vertical guess at most +-RY pixels),
// plus the previous row still in the pipeline and the next row being loaded: 2*RY+4 = NR rows, so RY = NR/2 - 2.
// Horizontally there is no limit (whole rows are buffered), so large horizontal motion costs no extra memory.
// Each cycle reads 4 pixels (2x2); the buffer is split into 4 banks by (row parity, column parity),
// so any 2x2 neighbourhood hits each bank exactly once -- each bank needs 1 read + 1 write port and fits in Block RAM.
//
// 11-stage pipeline that never stalls (for 250 MHz: no high-fanout "downstream not ready -> freeze everything" enable):
//   S0 accept a pixel -> S1 clamp -> S2 (x<<8)+gu -> S3 clip to the image, split integer/fraction
//   -> S4 bank addresses and RAM read -> S4b pick the byte from the 8-pixel word -> S5 place into 2x2, horizontal differences
//   -> S6 horizontal interpolation -> S7 vertical difference -> S8 vertical interpolation -> output FIFO (16 entries)
// Credit-based flow control: pixels accepted but not yet taken downstream (pipeline + FIFO) never exceed 16,
// so the FIFO can never overflow and the pipeline never needs to stall.
// Whether row ay may start and whether I1 may keep loading are precomputed from next-cycle register values (rows_ok_r).
// =============================================================================
`timescale 1ns/1ps

module lk_warp #(
    parameter WMAX = 1024,          // power of 2
    parameter NRB  = 6,             // ring buffer of 2^NRB rows (6 -> 64 rows, RY = 30; 5 -> 32 rows, RY = 14)
    parameter CW   = 16
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire [CW-1:0]        cfg_w,
    input  wire [CW-1:0]        cfg_h,

    input  wire                 b_valid,
    output wire                 b_ready,
    input  wire [7:0]           b_p1,

    input  wire                 a_valid,
    output wire                 a_ready,
    input  wire [7:0]           a_p0,
    input  wire signed [16:0]   a_gu2,
    input  wire signed [16:0]   a_gv2,

    output wire                 o_valid,
    input  wire                 o_ready,
    output wire [7:0]           o_p0,
    output wire [7:0]           o_p1w,
    output wire signed [15:0]   o_gu,
    output wire signed [15:0]   o_gv
);
    localparam AW  = $clog2(WMAX);
    localparam NR  = 1 << NRB;
    localparam RY  = NR / 2 - 2;
    localparam BAW = (NRB - 1) + (AW - 1);          // address width of each bank
    localparam signed [16:0] GXMAX = 17'sd32767;
    localparam signed [16:0] GYMAX = RY * 256;
    localparam PW  = CW + 10;                        // fixed-point coordinate width (signed)

    reg [CW-1:0] wm1, hm1;
    reg signed [PW-1:0] pxmax, pymax;
    always @(posedge clk) begin
        wm1   <= cfg_w - 1'b1;
        hm1   <= cfg_h - 1'b1;
        pxmax <= $signed({1'b0, cfg_w - 1'b1, 8'd0}) - 1;
        pymax <= $signed({1'b0, cfg_h - 1'b1, 8'd0}) - 1;
    end

    // ------------------------------------------------------------ 4 banks
    // Each bank word holds 8 pixels (64 bits) with byte writes, so a 32-row x 512-column bank is exactly
    // 1024 x 64 bits and fits one UltraRAM (URAM288, 4K x 72, byte-write capable) --
    // K26 has only 144 Block RAMs, which limit the lane count; it has 64 URAMs, and NRB=6 (64 rows) still fits in one.
    localparam WAW = BAW - 3;                        // word address width
    (* ram_style = "ultra" *) reg [63:0] bk_ee [0:(1<<WAW)-1];    // even row, even column
    (* ram_style = "ultra" *) reg [63:0] bk_eo [0:(1<<WAW)-1];    // even row, odd column
    (* ram_style = "ultra" *) reg [63:0] bk_oe [0:(1<<WAW)-1];
    (* ram_style = "ultra" *) reg [63:0] bk_oo [0:(1<<WAW)-1];

    // ------------------------------------------------------------ load I1 (b stream)
    reg [CW:0]   ld_row;                 // row being loaded (0..H, = number of rows loaded)
    reg [CW-1:0] ld_col;
    reg [CW:0]   ay;                     // output row being accepted (0..H)
    reg [CW-1:0] ax;

    // Row ld_row overwrites row ld_row-NR; it may only be written when that row is older than the earliest row
    // that may still be read (ay-1-RY): ld_row < ay + RY + 3 (precomputed one cycle early in b_ready_r below)
    reg    b_ready_r;
    assign b_ready = !rst && b_ready_r;
    wire   b_fire  = b_valid && b_ready;
    wire [BAW-1:0] wa = {ld_row[NRB-1:1], ld_col[AW-1:1]};

    always @(posedge clk) begin
        if (b_fire) begin
            case ({ld_row[0], ld_col[0]})
                2'b00: bk_ee[wa[BAW-1:3]][8*wa[2:0] +: 8] <= b_p1;
                2'b01: bk_eo[wa[BAW-1:3]][8*wa[2:0] +: 8] <= b_p1;
                2'b10: bk_oe[wa[BAW-1:3]][8*wa[2:0] +: 8] <= b_p1;
                2'b11: bk_oo[wa[BAW-1:3]][8*wa[2:0] +: 8] <= b_p1;
            endcase
        end
    end

    // ------------------------------------------------------------ flow control (credits + row readiness)
    // For 250 MHz every ready-type signal is a register, computed each cycle from the next-cycle counter values.
    // The +1 versions of the counters are registers too (ld_row1 = ld_row+1, lim1 = lim+1),
    // so the critical path is just a 2:1 mux and one compare, with no adder.
    localparam CAP = 16;
    reg  [4:0]  occ;                    // pixels accepted and not yet taken downstream (<= CAP)
    reg         rows_ok_r;              // all I1 rows needed by row ay are loaded
    reg         ay_done;                // all a pixels of this frame have been accepted
    reg  [CW:0] ld_row1;                // = ld_row + 1
    reg  [CW:0] lim, lim1;              // = ay + RY + 3, and that +1
    reg         ld_eolc, a_eolc;        // ld_col == W-1 / ax == W-1 (registered)
    wire        o_pop = o_valid && o_ready;

    assign a_ready = !rst && rows_ok_r && !ay_done && (occ < CAP);
    wire a_fire = a_valid && a_ready;

    // End of frame: all a pixels accepted and taken -> reset the counters and start loading the next frame
    wire drained = ay_done && (occ == 0);

    wire          ld_eol    = b_fire && ld_eolc;
    wire          a_eol     = a_fire && a_eolc;
    wire [CW:0]   ld_row_n  = drained ? {(CW+1){1'b0}} : (ld_eol ? ld_row1 : ld_row);
    wire [CW:0]   ld_row1_n = drained ? {{CW{1'b0}}, 1'b1} : (ld_eol ? ld_row1 + 1'b1 : ld_row1);
    wire [CW:0]   lim_n     = drained ? RY + 3 : (a_eol ? lim1 : lim);
    wire [CW:0]   lim1_n    = drained ? RY + 4 : (a_eol ? lim1 + 1'b1 : lim1);
    // Output row ay needs I1 row ay+RY+1 loaded: ld_row >= ay+RY+2 = lim-1  <=>  ld_row+1 >= lim
    wire          rows_ok_n = (ld_row_n >= {1'b0, cfg_h}) || (ld_row1_n >= lim_n);
    wire          b_rdy_n   = (ld_row_n < {1'b0, cfg_h}) && (ld_row_n < lim_n);
    wire [CW-1:0] ld_col_n  = drained ? {CW{1'b0}} : (b_fire ? (ld_eolc ? {CW{1'b0}} : ld_col + 1'b1) : ld_col);
    wire [CW-1:0] ax_n      = drained ? {CW{1'b0}} : (a_fire ? (a_eolc  ? {CW{1'b0}} : ax + 1'b1)     : ax);

    always @(posedge clk) begin
        if (rst) begin
            ld_row <= 0; ld_row1 <= 1; ld_col <= 0; ay <= 0; ax <= 0;
            lim <= RY + 3; lim1 <= RY + 4;
            occ <= 0; rows_ok_r <= 1'b0; ay_done <= 1'b0; b_ready_r <= 1'b0;
            ld_eolc <= 1'b0; a_eolc <= 1'b0;
        end else begin
            ld_row    <= ld_row_n;
            ld_row1   <= ld_row1_n;
            lim       <= lim_n;
            lim1      <= lim1_n;
            ld_col    <= ld_col_n;
            ax        <= ax_n;
            ld_eolc   <= (ld_col_n == wm1);
            a_eolc    <= (ax_n == wm1);
            rows_ok_r <= !drained && rows_ok_n;
            b_ready_r <= b_rdy_n;
            occ       <= occ + a_fire - o_pop;
            if (drained) begin
                ay <= 0; ay_done <= 1'b0;
            end else if (a_eol) begin
                ay <= ay + 1'b1;
                if (ay + 1'b1 == {1'b0, cfg_h}) ay_done <= 1'b1;
            end
        end
    end

    // ------------------------------------------------------------ pipeline registers
    reg s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v, s8_v;
    // S0
    reg [CW-1:0]        s0_x, s0_y;
    reg [7:0]           s0_p0;
    reg signed [16:0]   s0_gu2, s0_gv2;
    // S1
    reg [CW-1:0]        s1_x, s1_y;
    reg [7:0]           s1_p0;
    reg signed [15:0]   s1_gu, s1_gv;
    // S2
    reg [7:0]           s2_p0;
    reg signed [15:0]   s2_gu, s2_gv;
    reg signed [PW-1:0] s2_px, s2_py;
    // S3
    reg [7:0]           s3_p0;
    reg signed [15:0]   s3_gu, s3_gv;
    reg [CW-1:0]        s3_xi, s3_yi;
    reg [7:0]           s3_fx, s3_fy;
    // S4 (RAM output registers)
    reg [7:0]           s4_p0, s4_fx, s4_fy;
    reg signed [15:0]   s4_gu, s4_gv;
    reg                 s4_rt, s4_cl;       // is the upper row odd? is the left column odd?
    reg [63:0]          w_ee, w_eo, w_oe, w_oo; // words read from RAM (8 pixels each)
    reg [2:0]           b_ee, b_eo, b_oe, b_oo; // which byte to take
    // S4b: pick the byte out of the word
    reg                 s4b_v;
    reg [7:0]           s4b_p0, s4b_fx, s4b_fy;
    reg signed [15:0]   s4b_gu, s4b_gv;
    reg                 s4b_rt, s4b_cl;
    reg [7:0]           q_ee, q_eo, q_oe, q_oo;
    // S5
    reg [7:0]           s5_p0, s5_fx, s5_fy;
    reg signed [15:0]   s5_gu, s5_gv;
    reg [7:0]           i00, i10;
    reg signed [8:0]    d0, d1;             // horizontal differences of the upper/lower rows, I01-I00 and I11-I10
    // S6
    reg [7:0]           s6_p0, s6_fy;
    reg signed [15:0]   s6_gu, s6_gv;
    reg [15:0]          s6_top, s6_bot;
    // S7
    reg [7:0]           s7_p0, s7_fy;
    reg signed [15:0]   s7_gu, s7_gv;
    reg [24:0]          s7_base;            // top*256 + 32768
    reg signed [16:0]   s7_dv;              // bot - top
    // S8 = data written into the output FIFO
    reg [7:0]           s8_p0, s8_p1w;
    reg signed [15:0]   s8_gu, s8_gv;

    always @(posedge clk) begin
        if (rst) begin
            s0_v <= 0; s1_v <= 0; s2_v <= 0; s3_v <= 0; s4_v <= 0; s4b_v <= 0;
            s5_v <= 0; s6_v <= 0; s7_v <= 0; s8_v <= 0;
        end else begin
            s0_v <= a_fire; s1_v <= s0_v; s2_v <= s1_v; s3_v <= s2_v; s4_v <= s3_v;
            s4b_v <= s4_v; s5_v <= s4b_v;   s6_v <= s5_v; s7_v <= s6_v; s8_v <= s7_v;
        end
    end

    // S4 bank addresses
    wire [NRB-1:0] slot_t = s3_yi[NRB-1:0];
    wire [NRB-1:0] slot_b = slot_t + 1'b1;                  // natural wrap-around = ring
    wire [AW-1:0]  xi1    = s3_xi[AW-1:0] + 1'b1;
    wire [BAW-1:0] ra_ee  = {slot_b[NRB-1:1], xi1[AW-1:1]};          // even row = whichever of yi, yi+1 is even
    wire [BAW-1:0] ra_eo  = {slot_b[NRB-1:1], s3_xi[AW-1:1]};
    wire [BAW-1:0] ra_oe  = {slot_t[NRB-1:1], xi1[AW-1:1]};
    wire [BAW-1:0] ra_oo  = {slot_t[NRB-1:1], s3_xi[AW-1:1]};

    // S5: place the 4 bank outputs back into 2x2 positions
    wire [7:0] m00 = s4b_rt ? (s4b_cl ? q_oo : q_oe) : (s4b_cl ? q_eo : q_ee);   // top left
    wire [7:0] m01 = s4b_rt ? (s4b_cl ? q_oe : q_oo) : (s4b_cl ? q_ee : q_eo);   // top right
    wire [7:0] m10 = s4b_rt ? (s4b_cl ? q_eo : q_ee) : (s4b_cl ? q_oo : q_oe);   // bottom left
    wire [7:0] m11 = s4b_rt ? (s4b_cl ? q_ee : q_eo) : (s4b_cl ? q_oe : q_oo);   // bottom right

    // Bilinear arithmetic: top = I00*256 + (I01-I00)*fx, then (top*256 + (bot-top)*fy + 32768) >> 16
    wire signed [17:0] htop = $signed({2'b0, i00, 8'd0}) + d0 * $signed({1'b0, s5_fx});
    wire signed [17:0] hbot = $signed({2'b0, i10, 8'd0}) + d1 * $signed({1'b0, s5_fx});
    wire signed [26:0] vsum = $signed({2'b0, s7_base}) + s7_dv * $signed({1'b0, s7_fy});

    wire signed [PW-1:0] px_c = (s2_px < 0) ? {PW{1'b0}} : (s2_px > pxmax ? pxmax : s2_px);
    wire signed [PW-1:0] py_c = (s2_py < 0) ? {PW{1'b0}} : (s2_py > pymax ? pymax : s2_py);

    always @(posedge clk) begin
        // S0
        if (a_fire) begin
            s0_x <= ax;  s0_y <= ay[CW-1:0];
            s0_p0 <= a_p0;  s0_gu2 <= a_gu2;  s0_gv2 <= a_gv2;
        end
        // S1: clamp
        s1_x <= s0_x;  s1_y <= s0_y;  s1_p0 <= s0_p0;
        s1_gu <= (s0_gu2 > GXMAX) ? GXMAX[15:0] : (s0_gu2 < -GXMAX) ? -GXMAX[15:0] : s0_gu2[15:0];
        s1_gv <= (s0_gv2 > GYMAX) ? GYMAX[15:0] : (s0_gv2 < -GYMAX) ? -GYMAX[15:0] : s0_gv2[15:0];
        // S2: fixed-point coordinates
        s2_p0 <= s1_p0;  s2_gu <= s1_gu;  s2_gv <= s1_gv;
        s2_px <= $signed({2'b0, s1_x, 8'd0}) + s1_gu;
        s2_py <= $signed({2'b0, s1_y, 8'd0}) + s1_gv;
        // S3: clip to the image, split integer/fraction
        s3_p0 <= s2_p0;  s3_gu <= s2_gu;  s3_gv <= s2_gv;
        s3_xi <= px_c[CW+7:8];  s3_fx <= px_c[7:0];
        s3_yi <= py_c[CW+7:8];  s3_fy <= py_c[7:0];
        // S4: read the 4 banks
        s4_p0 <= s3_p0;  s4_gu <= s3_gu;  s4_gv <= s3_gv;
        s4_fx <= s3_fx;  s4_fy <= s3_fy;
        s4_rt <= s3_yi[0];  s4_cl <= s3_xi[0];
        w_ee <= bk_ee[ra_ee[BAW-1:3]];  b_ee <= ra_ee[2:0];
        w_eo <= bk_eo[ra_eo[BAW-1:3]];  b_eo <= ra_eo[2:0];
        w_oe <= bk_oe[ra_oe[BAW-1:3]];  b_oe <= ra_oe[2:0];
        w_oo <= bk_oo[ra_oo[BAW-1:3]];  b_oo <= ra_oo[2:0];
        // S4b: pick the bytes
        s4b_p0 <= s4_p0;  s4b_gu <= s4_gu;  s4b_gv <= s4_gv;
        s4b_fx <= s4_fx;  s4b_fy <= s4_fy;  s4b_rt <= s4_rt;  s4b_cl <= s4_cl;
        q_ee <= w_ee[8*b_ee +: 8];
        q_eo <= w_eo[8*b_eo +: 8];
        q_oe <= w_oe[8*b_oe +: 8];
        q_oo <= w_oo[8*b_oo +: 8];
        // S5: placement + horizontal differences
        s5_p0 <= s4b_p0;  s5_gu <= s4b_gu;  s5_gv <= s4b_gv;
        s5_fx <= s4b_fx;  s5_fy <= s4b_fy;
        i00 <= m00;  i10 <= m10;
        d0  <= $signed({1'b0, m01}) - $signed({1'b0, m00});
        d1  <= $signed({1'b0, m11}) - $signed({1'b0, m10});
        // S6: horizontal interpolation
        s6_p0 <= s5_p0;  s6_gu <= s5_gu;  s6_gv <= s5_gv;  s6_fy <= s5_fy;
        s6_top <= htop[15:0];
        s6_bot <= hbot[15:0];
        // S7: vertical difference
        s7_p0 <= s6_p0;  s7_gu <= s6_gu;  s7_gv <= s6_gv;  s7_fy <= s6_fy;
        s7_base <= {1'b0, s6_top, 8'd0} + 25'd32768;
        s7_dv   <= $signed({1'b0, s6_bot}) - $signed({1'b0, s6_top});
        // S8: vertical interpolation
        s8_p0 <= s7_p0;  s8_gu <= s7_gu;  s8_gv <= s7_gv;
        s8_p1w <= vsum[23:16];
    end

    // ------------------------------------------------------------ output FIFO (16 entries)
    (* ram_style = "distributed" *) reg [47:0] ofifo [0:CAP-1];
    reg [3:0]  owp, orp;
    reg [4:0]  ocnt;
    always @(posedge clk) begin
        if (s8_v) ofifo[owp] <= {s8_p0, s8_p1w, s8_gu, s8_gv};
        if (rst) begin
            owp <= 0; orp <= 0; ocnt <= 0;
        end else begin
            if (s8_v)  owp <= owp + 1'b1;
            if (o_pop) orp <= orp + 1'b1;
            ocnt <= ocnt + s8_v - o_pop;
        end
    end
    wire [47:0] ohead = ofifo[orp];
    assign o_valid = (ocnt != 0);
    assign o_p0    = ohead[47:40];
    assign o_p1w   = ohead[39:32];
    assign o_gu    = ohead[31:16];
    assign o_gv    = ohead[15:0];

endmodule
