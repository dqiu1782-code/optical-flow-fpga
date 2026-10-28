// =============================================================================
// tb_lk_top.v -- self-checking testbench for the complete LK pipeline
//
// Checks three places at once (each stage bit by bit, pixel by pixel against the Python golden answer):
//   (1) Stage 1 output Ix/Iy/It     (2) Stage 2 output, the 5 window sums     (3) final u/v/flag
// It also checks that nothing is dropped between Stage 1 and Stage 2, and counts clock cycles (to estimate FPGA speed).
//
// Run-time parameters:
//   +BUBBLE=30   probability (%) of upstream gaps, default 30; use 0 for speed measurements
//   +NFRAMES=2   number of frames fed back to back, default 2
//   +VCD         write the waveform lk_top.vcd (avoid for large images, the file gets huge)
// =============================================================================
`timescale 1ns/1ps

module tb_lk_top;

    `include "params.vh"            // IMG_W, IMG_H, DET_MIN
    localparam NPIX = IMG_W * IMG_H;

    wire [15:0] cfg_w = IMG_W;      // run-time size
    wire [15:0] cfg_h = IMG_H;
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst = 1'b1;

    reg         in_valid = 1'b0;
    wire        in_ready;
    reg  [7:0]  in_p0 = 8'd0, in_p1 = 8'd0;
    wire        out_valid, out_flag;
    wire [15:0] out_x, out_y;
    wire signed [15:0] out_u, out_v;

    lk_top #(.WMAX(WMAX), .CW(16), .DET_MIN(DET_MIN)) dut (
        .clk(clk), .rst(rst), .cfg_w(cfg_w), .cfg_h(cfg_h),
        .in_valid(in_valid), .in_ready(in_ready), .in_p0(in_p0), .in_p1(in_p1),
        .out_valid(out_valid), .out_x(out_x), .out_y(out_y),
        .out_u(out_u), .out_v(out_v), .out_flag(out_flag)
    );

    // ---------------- golden answers ----------------
    reg [7:0]  f0 [0:NPIX-1];
    reg [7:0]  f1 [0:NPIX-1];
    reg [8:0]  g_ix [0:NPIX-1];
    reg [8:0]  g_iy [0:NPIX-1];
    reg [8:0]  g_it [0:NPIX-1];
    reg [23:0] g_sxx [0:NPIX-1];
    reg [23:0] g_sxy [0:NPIX-1];
    reg [23:0] g_syy [0:NPIX-1];
    reg [23:0] g_sxt [0:NPIX-1];
    reg [23:0] g_syt [0:NPIX-1];
    reg [15:0] g_u [0:NPIX-1];
    reg [15:0] g_v [0:NPIX-1];
    reg [0:0]  g_flag [0:NPIX-1];

    integer i, frame, seed, bubble_pct, nframes;
    integer cycles = 0, first_in = -1, prev_end = -1;
    integer n1 = 0, n2 = 0, n3 = 0, e1 = 0, e2 = 0, e3 = 0, drops = 0, nflag = 0;
    integer idx;

    always @(posedge clk) cycles = cycles + 1;

    task report_err;
        input integer stage;
        input integer y;
        input integer x;
        begin
            if (e1 + e2 + e3 <= 12)
                $display("MISMATCH stage %0d at (y=%0d, x=%0d)", stage, y, x);
        end
    endtask

    // ---------------- stimulus ----------------
    initial begin
        $readmemh("vectors/frame0.hex", f0);
        $readmemh("vectors/frame1.hex", f1);
        $readmemh("vectors/golden_ix.hex", g_ix);
        $readmemh("vectors/golden_iy.hex", g_iy);
        $readmemh("vectors/golden_it.hex", g_it);
        $readmemh("vectors/golden_sxx.hex", g_sxx);
        $readmemh("vectors/golden_sxy.hex", g_sxy);
        $readmemh("vectors/golden_syy.hex", g_syy);
        $readmemh("vectors/golden_sxt.hex", g_sxt);
        $readmemh("vectors/golden_syt.hex", g_syt);
        $readmemh("vectors/golden_u.hex", g_u);
        $readmemh("vectors/golden_v.hex", g_v);
        $readmemh("vectors/golden_flag.hex", g_flag);
        if (!$value$plusargs("BUBBLE=%d", bubble_pct)) bubble_pct = 30;
        if (!$value$plusargs("NFRAMES=%d", nframes))   nframes = 2;
        seed = 12345;
        if ($test$plusargs("VCD")) begin
            $dumpfile("lk_top.vcd");
            $dumpvars(0, tb_lk_top);
        end

        repeat (3) @(posedge clk);
        rst <= 1'b0;

        for (frame = 0; frame < nframes; frame = frame + 1) begin
            i = 0;
            while (i < NPIX) begin
                in_valid <= ((($random(seed) % 100) + 100) % 100) >= bubble_pct;
                in_p0    <= f0[i];
                in_p1    <= f1[i];
                @(posedge clk);
                if (in_valid && in_ready) begin
                    if (first_in < 0) first_in = cycles;
                    i = i + 1;
                end
            end
        end
        in_valid <= 1'b0;
    end

    // ---------------- check (1) Stage 1 ----------------
    always @(posedge clk) if (dut.g_valid) begin
        idx = dut.g_y * IMG_W + dut.g_x;
        if (idx !== n1 % NPIX || dut.g_ix !== g_ix[idx] || dut.g_iy !== g_iy[idx] || dut.g_it !== g_it[idx]) begin
            e1 = e1 + 1; report_err(1, dut.g_y, dut.g_x);
        end
        n1 = n1 + 1;
        if (!dut.w_in_ready) drops = drops + 1;     // data arriving while Stage 2 is not ready = dropped data
    end

    // ---------------- check (2) Stage 2 ----------------
    always @(posedge clk) if (dut.w_valid) begin
        idx = dut.w_y * IMG_W + dut.w_x;
        if (idx !== n2 % NPIX || dut.w_sxx !== g_sxx[idx] || dut.w_sxy !== g_sxy[idx] || dut.w_syy !== g_syy[idx]
            || dut.w_sxt !== g_sxt[idx] || dut.w_syt !== g_syt[idx]) begin
            e2 = e2 + 1; report_err(2, dut.w_y, dut.w_x);
            if (e2 <= 3)
                $display("   got Sxx=%0d Sxy=%0d Syy=%0d Sxt=%0d Syt=%0d | exp %0d %0d %0d %0d %0d",
                         dut.w_sxx, dut.w_sxy, dut.w_syy, dut.w_sxt, dut.w_syt,
                         $signed(g_sxx[idx]), $signed(g_sxy[idx]), $signed(g_syy[idx]),
                         $signed(g_sxt[idx]), $signed(g_syt[idx]));
        end
        n2 = n2 + 1;
    end

    // ---------------- check (3) final output ----------------
    always @(posedge clk) if (out_valid) begin
        idx = out_y * IMG_W + out_x;
        if (idx !== n3 % NPIX || out_u !== g_u[idx] || out_v !== g_v[idx] || out_flag !== g_flag[idx]) begin
            e3 = e3 + 1; report_err(3, out_y, out_x);
            if (e3 <= 3)
                $display("   got u=%0d v=%0d flag=%0d | exp u=%0d v=%0d flag=%0d",
                         out_u, out_v, out_flag, $signed(g_u[idx]), $signed(g_v[idx]), g_flag[idx]);
        end
        if (out_flag) nflag = nflag + 1;
        if (out_x == IMG_W-1 && out_y == IMG_H-1) begin
            if (prev_end < 0)
                $display("Frame 1 latency: %0d cycles from first pixel in to last flow vector out", cycles - first_in + 1);
            else
                $display("Steady-state: %0d cycles per frame (frame-end to frame-end)", cycles - prev_end);
            prev_end = cycles;
        end
        n3 = n3 + 1;
    end

    // ---------------- end ----------------
    initial begin : finish_block
        wait (rst == 1'b0);
        wait (n3 == NPIX * nframes);
        repeat (2) @(posedge clk);
        $display("------------------------------------------------------------");
        $display("Image %0dx%0d, %0d frames, bubble=%0d%%, DET_MIN=%0d", IMG_W, IMG_H, nframes, bubble_pct, DET_MIN);
        $display("Stage1 outputs %0d errors %0d | Stage2 outputs %0d errors %0d | Final outputs %0d errors %0d",
                 n1, e1, n2, e2, n3, e3);
        $display("Dropped transfers: %0d   Pixels with valid flow (flag=1): %0d", drops, nflag);
        if (e1 == 0 && e2 == 0 && e3 == 0 && drops == 0 && n1 == NPIX*nframes && n2 == NPIX*nframes)
            $display("*** PASS ***");
        else
            $display("*** FAIL ***");
        $display("------------------------------------------------------------");
        $finish;
    end

    initial begin
        #(64'd10 * NPIX * 40 * 20);
        $display("*** FAIL: TIMEOUT (stage outputs %0d / %0d / %0d) ***", n1, n2, n3);
        $finish;
    end

endmodule
