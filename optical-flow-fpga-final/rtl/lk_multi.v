// =============================================================================
// lk_multi.v  --  several LK pipelines in parallel: LANES frame pairs at the same time
//
// Each lane is a complete lk_top with its own input and output streams, independent of the others.
// LANES pairs go in and come out together; throughput = LANES x one pipeline.
// How many lanes fit is set by two things: chip resources (LUT / DSP / on-chip memory) and DDR bandwidth.
// =============================================================================
`timescale 1ns/1ps

module lk_multi #(
    parameter LANES = 4,
    parameter WMAX  = 1024,
    parameter CW    = 16,
    parameter [63:0] DET_MIN = 64'd65536
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire [CW-1:0]            cfg_w,      // all lanes share one size
    input  wire [CW-1:0]            cfg_h,

    input  wire [LANES-1:0]         in_valid,
    output wire [LANES-1:0]         in_ready,
    input  wire [8*LANES-1:0]       in_p0,      // lane l's pixel is in [8*l +: 8]
    input  wire [8*LANES-1:0]       in_p1,

    output wire [LANES-1:0]         out_valid,
    output wire [CW*LANES-1:0]      out_x,
    output wire [CW*LANES-1:0]      out_y,
    output wire [16*LANES-1:0]      out_u,
    output wire [16*LANES-1:0]      out_v,
    output wire [LANES-1:0]         out_flag
);

    genvar l;
    generate
        for (l = 0; l < LANES; l = l + 1) begin : lane
            lk_top #(.WMAX(WMAX), .CW(CW), .DET_MIN(DET_MIN)) u_lk (
                .clk(clk), .rst(rst), .cfg_w(cfg_w), .cfg_h(cfg_h),
                .in_valid(in_valid[l]), .in_ready(in_ready[l]),
                .in_p0(in_p0[8*l +: 8]), .in_p1(in_p1[8*l +: 8]),
                .out_valid(out_valid[l]),
                .out_x(out_x[CW*l +: CW]), .out_y(out_y[CW*l +: CW]),
                .out_u(out_u[16*l +: 16]), .out_v(out_v[16*l +: 16]),
                .out_flag(out_flag[l])
            );
        end
    endgenerate

endmodule
