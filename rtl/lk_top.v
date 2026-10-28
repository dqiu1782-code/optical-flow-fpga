// =============================================================================
// lk_top.v  --  complete Lucas-Kanade optical flow pipeline
//
//   two pixel streams --> [Stage 1 gradient] --> [Stage 2 5x5 window] --> [Stage 3 solve] --> (u, v)
//
// Steady-state throughput: 1 pixel per clock. A W x H frame takes about W*H + 3W cycles
// (the extra 3W is the FLUSH of two stages: border rows emitted at the end of a frame).
// v2: image size is a run-time input cfg_w/cfg_h (up to WMAX); stages 2 and 3 gained pipeline registers.
//
// Handshake: pixels are accepted from upstream only when both Stage 1 and Stage 2 can take data.
//       Stage 2 accepts nothing during its end-of-frame FLUSH, so the pipeline pauses before the next frame.
// =============================================================================
`timescale 1ns/1ps

module lk_top #(
    parameter WMAX = 1024,          // maximum supported width (line-buffer size)
    parameter CW   = 16,
    parameter [63:0] DET_MIN = 64'd65536
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire [CW-1:0]        cfg_w,      // run-time image size (differs per pyramid level)
    input  wire [CW-1:0]        cfg_h,

    input  wire                 in_valid,
    output wire                 in_ready,
    input  wire [7:0]           in_p0,      // pixel of frame t
    input  wire [7:0]           in_p1,      // pixel of frame t+1

    output wire                 out_valid,
    output wire [CW-1:0]        out_x,
    output wire [CW-1:0]        out_y,
    output wire signed [15:0]   out_u,      // Q8: pixels = out_u / 256
    output wire signed [15:0]   out_v,
    output wire                 out_flag
);

    // ---------------- Stage 1 ----------------
    wire                g_in_ready, g_valid, g_done;
    wire [CW-1:0]       g_x, g_y;
    wire signed [8:0]   g_ix, g_iy, g_it;

    // ---------------- Stage 2 ----------------
    wire                w_in_ready, w_valid, w_done;
    wire [CW-1:0]       w_x, w_y;
    wire signed [23:0]  w_sxx, w_sxy, w_syy, w_sxt, w_syt;

    assign in_ready = g_in_ready & w_in_ready;

    lk_gradient #(.WMAX(WMAX), .CW(CW)) u_grad (
        .clk(clk), .rst(rst), .cfg_w(cfg_w), .cfg_h(cfg_h),
        .in_valid(in_valid & w_in_ready), .in_ready(g_in_ready),
        .in_p0(in_p0), .in_p1(in_p1),
        .out_valid(g_valid), .out_x(g_x), .out_y(g_y),
        .out_ix(g_ix), .out_iy(g_iy), .out_it(g_it),
        .frame_done(g_done)
    );

    lk_window #(.WMAX(WMAX), .CW(CW), .SW(24)) u_win (
        .clk(clk), .rst(rst), .cfg_w(cfg_w), .cfg_h(cfg_h),
        .in_valid(g_valid), .in_ready(w_in_ready),
        .in_ix(g_ix), .in_iy(g_iy), .in_it(g_it),
        .out_valid(w_valid), .out_x(w_x), .out_y(w_y),
        .out_sxx(w_sxx), .out_sxy(w_sxy), .out_syy(w_syy), .out_sxt(w_sxt), .out_syt(w_syt),
        .frame_done(w_done)
    );

    // ---------------- Stage 3 ----------------
    lk_solve #(.CW(CW), .SW(24), .DET_MIN(DET_MIN), .FRAC(8), .QB(15)) u_solve (
        .clk(clk), .rst(rst),
        .in_valid(w_valid), .in_x(w_x), .in_y(w_y),
        .in_sxx(w_sxx), .in_sxy(w_sxy), .in_syy(w_syy), .in_sxt(w_sxt), .in_syt(w_syt),
        .out_valid(out_valid), .out_x(out_x), .out_y(out_y),
        .out_u(out_u), .out_v(out_v), .out_flag(out_flag)
    );

endmodule
