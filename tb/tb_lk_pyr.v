// =============================================================================
// tb_lk_pyr.v -- self-checking testbench for the whole pyramid LK chain
//
// The testbench plays "DMA + scheduler" (on the board this is the driver on the ARM):
//   Step 1: stream both frames into lk_pyramid, store every output level in "DDR" (arrays in the TB),
//           and compare pixel by pixel with the Python pyramid.
//   Step 2: from the coarsest level to full resolution, for each level:
//           set the size -> feed I0 and I1 at this level and the coarser-level flow (the hardware's own previous output)
//           -> collect this level's flow into "DDR" and compare bit by bit with Python.
//           The warp output and the lk_top output are compared too, to localise any error.
//   Step 3: at full resolution only the statistics registers are checked, bit by bit against Python.
//
// Run-time parameter: +BUBBLE=20 (percentage of random gaps in each input stream; 0 for speed measurements)
// In params.vh: DU_MAX = per-level residual clamp (0 = off), NRB = log2 of the warp ring-buffer rows
// =============================================================================
`timescale 1ns/1ps

module tb_lk_pyr;

    `include "params.vh"        // IMG_W IMG_H LEVELS NRB WMAX TOT DET_MIN lvl_w() lvl_h() lvl_off()
    localparam NPIX = IMG_W * IMG_H;

    reg clk = 1'b0;
    always #2 clk = ~clk;
    reg rst = 1'b1;
    integer cycles = 0;
    always @(posedge clk) cycles = cycles + 1;

    // ------------------------------------------------------------ pyramid generator
    wire [15:0] img_w = IMG_W;
    wire [15:0] img_h = IMG_H;
    reg         pb_valid = 1'b0;
    wire        pb_ready;
    reg  [7:0]  pb_p0 = 0, pb_p1 = 0;
    wire [LEVELS-1:0]   lv_valid;
    wire [8*LEVELS-1:0] lv_p0, lv_p1;
    wire [39:0]         st_sharp;
    wire                st_sharp_done;
    integer             sharp_err = 0;

    lk_pyramid #(.LEVELS(LEVELS), .WMAX(WMAX), .CW(16)) u_pyr (
        .clk(clk), .rst(rst), .cfg_w(img_w), .cfg_h(img_h),
        .in_valid(pb_valid), .in_ready(pb_ready), .in_p0(pb_p0), .in_p1(pb_p1),
        .lv_valid(lv_valid), .lv_p0(lv_p0), .lv_p1(lv_p1),
        .st_sharp(st_sharp), .st_sharp_done(st_sharp_done));

    // ------------------------------------------------------------ per-level engine
    reg  [15:0] cfg_w = IMG_W, cfg_h = IMG_H;
    wire [14:0] dumax = DU_MAX;
    reg         cfg_guess = 1'b0, cfg_stats = 1'b0, stats_clear = 1'b0;
    reg         p_valid = 1'b0, b_valid = 1'b0, c_valid = 1'b0;
    wire        p_ready, b_ready, c_ready;
    reg  [7:0]  p_p0 = 0, b_p1 = 0;
    reg  signed [15:0] c_u = 0, c_v = 0;
    wire        f_valid, f_flag;
    wire [15:0] f_x, f_y;
    wire signed [15:0] f_u, f_v;
    wire [23:0] st_count;
    wire signed [39:0] st_sum_u, st_sum_v;
    wire [39:0] st_sum_l1;
    wire [16:0] st_max_l1;
    wire [8*24-1:0] st_hist;

    lk_level #(.WMAX(WMAX), .NRB(NRB), .CW(16), .DET_MIN(DET_MIN)) dut (
        .clk(clk), .rst(rst), .cfg_w(cfg_w), .cfg_h(cfg_h),
        .cfg_guess(cfg_guess), .cfg_stats(cfg_stats), .cfg_dumax(dumax), .stats_clear(stats_clear),
        .p_valid(p_valid), .p_ready(p_ready), .p_p0(p_p0),
        .b_valid(b_valid), .b_ready(b_ready), .b_p1(b_p1),
        .c_valid(c_valid), .c_ready(c_ready), .c_u(c_u), .c_v(c_v),
        .f_valid(f_valid), .f_x(f_x), .f_y(f_y), .f_u(f_u), .f_v(f_v), .f_flag(f_flag),
        .st_count(st_count), .st_sum_u(st_sum_u), .st_sum_v(st_sum_v), .st_sum_l1(st_sum_l1),
        .st_max_l1(st_max_l1), .st_hist(st_hist));

    // ------------------------------------------------------------ "DDR" and the golden answers
    reg [7:0]  f0  [0:NPIX-1];
    reg [7:0]  f1  [0:NPIX-1];
    reg [7:0]  m0  [0:TOT-1];       // pyramid produced by the hardware
    reg [7:0]  m1  [0:TOT-1];
    reg [15:0] mu  [0:TOT-1];       // per-level flow produced by the hardware
    reg [15:0] mv  [0:TOT-1];
    reg [7:0]  gp0 [0:TOT-1];
    reg [7:0]  gp1 [0:TOT-1];
    reg [7:0]  gw  [0:TOT-1];
    reg [15:0] gdu [0:TOT-1];
    reg [15:0] gdv [0:TOT-1];
    reg [15:0] gfu [0:TOT-1];
    reg [15:0] gfv [0:TOT-1];
    reg [0:0]  gff [0:TOT-1];
    reg [39:0] gst [0:12];

    integer bubble_pct, seed_p, seed_b, seed_c, seed_pb;
    integer i, k, l, n, off, offc, nc;
    integer cur_lvl = -1, cur_off = 0, cur_w = 0;
    integer pyr_err = 0, pyr_cnt = 0, pyr_cycles = 0;
    integer n_f = 0, e_f = 0, n_w = 0, e_w = 0, n_d = 0, e_d = 0, nflag = 0;
    integer tot_e_f = 0, tot_e_w = 0, tot_e_d = 0, st_err = 0, lvl_cycles = 0, all_cycles = 0;
    integer t0, idx;
    integer lvcnt [0:LEVELS-1];

    // ------------------------------------------------------------ step 1: collect the pyramid levels
    genvar gl;
    generate
        for (gl = 1; gl < LEVELS; gl = gl + 1) begin : cap
            always @(posedge clk) if (lv_valid[gl]) begin
                m0[lvl_off(gl) + lvcnt[gl]] = lv_p0[8*gl +: 8];
                m1[lvl_off(gl) + lvcnt[gl]] = lv_p1[8*gl +: 8];
                lvcnt[gl] = lvcnt[gl] + 1;
            end
        end
    endgenerate

    // ------------------------------------------------------------ monitors (step 2)
    // warp output: compared with Python's I1w (counted in order)
    always @(posedge clk) if (cur_lvl >= 0 && dut.w_valid && dut.w_ready) begin
        if (dut.w_p1w !== gw[cur_off + n_w] || dut.w_p0 !== gp0[cur_off + n_w]) begin
            e_w = e_w + 1;
            if (e_w <= 3) $display("  WARP MISMATCH level %0d pixel %0d: got p0=%0d i1w=%0d exp p0=%0d i1w=%0d",
                                   cur_lvl, n_w, dut.w_p0, dut.w_p1w, gp0[cur_off + n_w], gw[cur_off + n_w]);
        end
        n_w = n_w + 1;
    end
    // lk_top output (residual du, dv)
    always @(posedge clk) if (cur_lvl >= 0 && dut.l_valid) begin
        idx = dut.l_y * cur_w + dut.l_x;
        if (idx !== n_d || dut.l_u !== gdu[cur_off + idx] || dut.l_v !== gdv[cur_off + idx]) begin
            e_d = e_d + 1;
            if (e_d <= 3) $display("  DU MISMATCH level %0d (y=%0d,x=%0d): got %0d,%0d exp %0d,%0d",
                                   cur_lvl, dut.l_y, dut.l_x, dut.l_u, dut.l_v,
                                   $signed(gdu[cur_off + idx]), $signed(gdv[cur_off + idx]));
        end
        n_d = n_d + 1;
    end
    // final flow u = g + du: written back to "DDR" and compared
    always @(posedge clk) if (cur_lvl >= 0 && f_valid) begin
        idx = f_y * cur_w + f_x;
        mu[cur_off + idx] = f_u;
        mv[cur_off + idx] = f_v;
        if (idx !== n_f || f_u !== gfu[cur_off + idx] || f_v !== gfv[cur_off + idx] || f_flag !== gff[cur_off + idx]) begin
            e_f = e_f + 1;
            if (e_f <= 3) $display("  FLOW MISMATCH level %0d (y=%0d,x=%0d): got %0d,%0d,%0d exp %0d,%0d,%0d",
                                   cur_lvl, f_y, f_x, f_u, f_v, f_flag,
                                   $signed(gfu[cur_off + idx]), $signed(gfv[cur_off + idx]), gff[cur_off + idx]);
        end
        if (f_flag) nflag = nflag + 1;
        n_f = n_f + 1;
    end

    // ------------------------------------------------------------ drivers of the three input streams
    task drive_p; input integer o; input integer cnt; integer j; begin
        j = 0;
        while (j < cnt) begin
            p_valid <= ((($random(seed_p) % 100) + 100) % 100) >= bubble_pct;
            p_p0    <= m0[o + j];
            @(posedge clk);
            if (p_valid && p_ready) j = j + 1;
        end
        p_valid <= 1'b0;
    end endtask

    task drive_b; input integer o; input integer cnt; integer j; begin
        j = 0;
        while (j < cnt) begin
            b_valid <= ((($random(seed_b) % 100) + 100) % 100) >= bubble_pct;
            b_p1    <= m1[o + j];
            @(posedge clk);
            if (b_valid && b_ready) j = j + 1;
        end
        b_valid <= 1'b0;
    end endtask

    task drive_c; input integer o; input integer cnt; integer j; begin
        j = 0;
        while (j < cnt) begin
            c_valid <= ((($random(seed_c) % 100) + 100) % 100) >= bubble_pct;
            c_u     <= mu[o + j];
            c_v     <= mv[o + j];
            @(posedge clk);
            if (c_valid && c_ready) j = j + 1;
        end
        c_valid <= 1'b0;
    end endtask

    // ------------------------------------------------------------ main sequence
    initial begin
        $readmemh("vectors/frame0.hex", f0);
        $readmemh("vectors/frame1.hex", f1);
        $readmemh("vectors/golden_pyr0.hex", gp0);
        $readmemh("vectors/golden_pyr1.hex", gp1);
        $readmemh("vectors/golden_i1w.hex", gw);
        $readmemh("vectors/golden_du.hex", gdu);
        $readmemh("vectors/golden_dv.hex", gdv);
        $readmemh("vectors/golden_fu.hex", gfu);
        $readmemh("vectors/golden_fv.hex", gfv);
        $readmemh("vectors/golden_ff.hex", gff);
        $readmemh("vectors/golden_stats.hex", gst);
        if (!$value$plusargs("BUBBLE=%d", bubble_pct)) bubble_pct = 20;
        seed_p = 11; seed_b = 22; seed_c = 33; seed_pb = 44;
        for (k = 0; k < LEVELS; k = k + 1) lvcnt[k] = 0;
        for (k = 0; k < NPIX; k = k + 1) begin m0[k] = f0[k]; m1[k] = f1[k]; end   // level 0 = the input image

        repeat (4) @(posedge clk);
        rst <= 1'b0;
        repeat (2) @(posedge clk);

        // ---------------- step 1: build the pyramid ----------------
        t0 = cycles;
        i = 0;
        while (i < NPIX) begin
            pb_valid <= ((($random(seed_pb) % 100) + 100) % 100) >= bubble_pct;
            pb_p0 <= f0[i];
            pb_p1 <= f1[i];
            @(posedge clk);
            if (pb_valid && pb_ready) i = i + 1;
        end
        pb_valid <= 1'b0;
        repeat (4) @(posedge clk);
        pyr_cycles = cycles - t0;
        for (l = 1; l < LEVELS; l = l + 1) begin
            n = lvl_w(l) * lvl_h(l);
            off = lvl_off(l);
            if (lvcnt[l] != n) begin
                pyr_err = pyr_err + 1;
                $display("  PYR level %0d: got %0d pixels, expected %0d", l, lvcnt[l], n);
            end
            for (k = 0; k < n; k = k + 1)
                if (m0[off + k] !== gp0[off + k] || m1[off + k] !== gp1[off + k]) begin
                    if (pyr_err < 3) $display("  PYR MISMATCH level %0d pixel %0d: got %0d,%0d exp %0d,%0d",
                                              l, k, m0[off + k], m1[off + k], gp0[off + k], gp1[off + k]);
                    pyr_err = pyr_err + 1;
                end
            pyr_cnt = pyr_cnt + n;
        end
        $display("PYRAMID %0d levels from %0dx%0d: %0d pixels checked, errors %0d, cycles %0d",
                 LEVELS, IMG_W, IMG_H, pyr_cnt, pyr_err, pyr_cycles);
        repeat (4) @(posedge clk);      // lk_sharp pipeline is 5 cycles; wait for the last pixel to accumulate
        if (st_sharp !== EXP_SHARP) sharp_err = 1;
        $display("SHARP %0d expected %0d %s", st_sharp, EXP_SHARP, sharp_err ? "MISMATCH" : "ok");

        // ---------------- step 2: level by level, coarse to fine ----------------
        for (l = LEVELS - 1; l >= 0; l = l - 1) begin
            n   = lvl_w(l) * lvl_h(l);
            off = lvl_off(l);
            cfg_w <= lvl_w(l);
            cfg_h <= lvl_h(l);
            cfg_guess <= (l != LEVELS - 1);
            cfg_stats <= (l == 0);
            stats_clear <= (l == 0);
            @(posedge clk);
            stats_clear <= 1'b0;
            repeat (3) @(posedge clk);          // let the configuration registers take effect
            n_f = 0; e_f = 0; n_w = 0; e_w = 0; n_d = 0; e_d = 0; nflag = 0;
            cur_off = off; cur_w = lvl_w(l); cur_lvl = l;
            t0 = cycles;
            if (l != LEVELS - 1) begin
                offc = lvl_off(l + 1);
                nc   = lvl_w(l + 1) * lvl_h(l + 1);
                fork
                    drive_p(off, n);
                    drive_b(off, n);
                    drive_c(offc, nc);
                join
            end else begin
                fork
                    drive_p(off, n);
                    drive_b(off, n);
                join
            end
            wait (n_f == n);
            lvl_cycles = cycles - t0;
            all_cycles = all_cycles + lvl_cycles;
            @(posedge clk);
            cur_lvl = -1;
            $display("LEVEL %0d %0dx%0d: flow outputs %0d errors %0d | warp errors %0d | du errors %0d | flag=1 %0d | cycles %0d | fifo max %0d",
                     l, lvl_w(l), lvl_h(l), n_f, e_f, e_w, e_d, nflag, lvl_cycles, dut.g_max);
            tot_e_f = tot_e_f + e_f + (n_w != n) + (n_d != n);
            tot_e_w = tot_e_w + e_w;
            tot_e_d = tot_e_d + e_d;
            repeat (4) @(posedge clk);
        end

        // ---------------- step 3: statistics registers ----------------
        repeat (8) @(posedge clk);
        if (st_count   !== gst[0][23:0]) st_err = st_err + 1;
        if (st_sum_u   !== gst[1])       st_err = st_err + 1;
        if (st_sum_v   !== gst[2])       st_err = st_err + 1;
        if (st_sum_l1  !== gst[3])       st_err = st_err + 1;
        if (st_max_l1  !== gst[4][16:0]) st_err = st_err + 1;
        for (k = 0; k < 8; k = k + 1)
            if (st_hist[24*k +: 24] !== gst[5 + k][23:0]) st_err = st_err + 1;
        $display("STATS count %0d sum_u %0d sum_v %0d sum_l1 %0d max_l1 %0d | expected count %0d sum_u %0d sum_v %0d sum_l1 %0d max_l1 %0d | mismatches %0d",
                 st_count, st_sum_u, st_sum_v, st_sum_l1, st_max_l1,
                 gst[0], $signed(gst[1]), $signed(gst[2]), gst[3], gst[4], st_err);
        $display("HIST %0d %0d %0d %0d %0d %0d %0d %0d", st_hist[0 +: 24], st_hist[24 +: 24], st_hist[48 +: 24],
                 st_hist[72 +: 24], st_hist[96 +: 24], st_hist[120 +: 24], st_hist[144 +: 24], st_hist[168 +: 24]);
        $display("Cycles: pyramid build %0d, all levels %0d (bubble=%0d%%)", pyr_cycles, all_cycles, bubble_pct);
        $display("------------------------------------------------------------");
        if (pyr_err == 0 && tot_e_f == 0 && tot_e_w == 0 && tot_e_d == 0 && st_err == 0 && sharp_err == 0)
            $display("*** PASS ***");
        else
            $display("*** FAIL *** (pyr %0d, flow %0d, warp %0d, du %0d, stats %0d, sharp %0d)",
                     pyr_err, tot_e_f, tot_e_w, tot_e_d, st_err, sharp_err);
        $finish;
    end

    initial begin
        #(64'd4 * TOT * 60 + 64'd4 * 200000);
        $display("*** FAIL: TIMEOUT (level %0d, flow outputs %0d, warp outputs %0d) ***", cur_lvl, n_f, n_w);
        $finish;
    end

endmodule
