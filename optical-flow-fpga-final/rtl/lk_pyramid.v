// =============================================================================
// lk_pyramid.v  --  image pyramid generator (streaming)
//
// The image streams in raster order and levels 1..LEVELS-1 come out at the same time (each halves width and height):
//     I_{l+1}(y, x) = (a + b + c + d + 2) >> 2       a..d = the 2x2 block of I_l
// Sizes round down (the last column/row is dropped for odd sizes), matching downsample in python/lk_pyramid.py.
//
// Each level is one lk_down, chained: the output stream of level 1 is the input stream of level 2.
// Level l outputs at 1/4^l of the input rate, so the whole pyramid is built in the same W*H cycles as the image itself.
// The DMA writes the levels back to DDR and reads them again level by level for the flow (the testbench plays the DMA here).
//
// Each frame needs its pyramid built only once: the pyramid of frame t is the "second frame" of pair (t-1, t)
// and the "first frame" of pair (t, t+1).
// =============================================================================
`timescale 1ns/1ps

module lk_down #(
    parameter WMAX = 1024,          // maximum input width
    parameter CW   = 16
)(
    input  wire             clk,
    input  wire             rst,
    input  wire [CW-1:0]    cfg_w,      // input image size
    input  wire [CW-1:0]    cfg_h,
    input  wire             in_valid,   // always accepted (in_ready is tied to 1)
    input  wire [7:0]       in_p,
    output reg              out_valid,
    output reg  [7:0]       out_p
);
    localparam HB = (WMAX > 2) ? $clog2(WMAX / 2) : 1;

    reg [CW-1:0] col, row, wm1, hm1, wlim, hlim;
    reg [7:0]    hold;                  // pixel of the even column
    reg [8:0]    q_up;                  // horizontal pair sum at the same position of the previous (even) row (synchronous read)

    always @(posedge clk) begin
        wm1  <= cfg_w - 1'b1;
        hm1  <= cfg_h - 1'b1;
        wlim <= cfg_w >> 1;             // output width
        hlim <= cfg_h >> 1;
    end

    wire [8:0]    hsum   = {1'b0, hold} + {1'b0, in_p};
    wire [CW-1:0] cx     = col >> 1;
    wire          col_ok = col[0] && (cx < wlim);       // odd column and the pair is complete
    wire          row_ok = (row >> 1) < hlim;
    wire [9:0]    total  = {1'b0, q_up} + {1'b0, hsum} + 10'd2;

    // Line buffer of the even row's horizontal pair sums. Level 1 (512 entries) goes to Block RAM, smaller levels to LUT RAM
    generate
        if (WMAX >= 1024) begin : g_bram
            (* ram_style = "block" *) reg [8:0] rowbuf [0:WMAX/2-1];
            always @(posedge clk) begin
                if (in_valid) begin
                    if (!col[0]) q_up <= rowbuf[cx[HB-1:0]];             // prefetch on even columns
                    if (col_ok && !row[0]) rowbuf[cx[HB-1:0]] <= hsum;   // even row: store the horizontal sum
                end
            end
        end else begin : g_lutram
            (* ram_style = "distributed" *) reg [8:0] rowbuf [0:WMAX/2-1];
            always @(posedge clk) begin
                if (in_valid) begin
                    if (!col[0]) q_up <= rowbuf[cx[HB-1:0]];
                    if (col_ok && !row[0]) rowbuf[cx[HB-1:0]] <= hsum;
                end
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (rst) begin
            col <= 0; row <= 0; hold <= 0;
            out_valid <= 1'b0; out_p <= 0;
        end else begin
            out_valid <= 1'b0;
            if (in_valid) begin
                if (!col[0]) hold <= in_p;
                if (col_ok && row[0] && row_ok) begin
                    out_valid <= 1'b1;
                    out_p     <= total[9:2];
                end
                if (col == wm1) begin
                    col <= 0;
                    row <= (row == hm1) ? {CW{1'b0}} : row + 1'b1;
                end else
                    col <= col + 1'b1;
            end
        end
    end
endmodule

// =============================================================================
// lk_sharp: per-frame sharpness statistic (on the level-0 I0 input stream, done while the pyramid is built, no extra cycles)
//     E = Σ_{y>=1,x>=1} (I[y][x]-I[y][x-1])² + (I[y][x]-I[y-1][x])²
// Motion blur weakens fine-scale gradients and lowers E; the ARM picks the frame with the largest E in each candidate window.
// Resources: one 8-bit line buffer (1024 entries = half a BRAM36) + 2 9x9 multipliers + one 40-bit accumulator.
// Cleared automatically on the first pixel of a frame; st_sharp holds the previous frame's result until the next frame starts.
// =============================================================================
module lk_sharp #(
    parameter WMAX = 1024,
    parameter CW   = 16,
    parameter SW   = 40
)(
    input  wire             clk,
    input  wire             rst,
    input  wire [CW-1:0]    cfg_w,
    input  wire [CW-1:0]    cfg_h,
    input  wire             in_valid,
    input  wire [7:0]       in_p,
    output reg  [SW-1:0]    st_sharp,
    output reg              st_done         // pulses for one cycle after the last pixel of a frame is accumulated
);
    localparam AB = $clog2(WMAX);
    reg [CW-1:0] col, row, wm1, hm1;
    always @(posedge clk) begin
        wm1 <= cfg_w - 1'b1;
        hm1 <= cfg_h - 1'b1;
    end
    (* ram_style = "block" *) reg [7:0] lb [0:WMAX-1];
    // S1: read the same column of the previous row (synchronous, read-first); register the current and left pixels
    reg        s1_v, s1_first, s1_last, s1_hx, s1_hy;
    reg [7:0]  s1_p, s1_left, s1_up;
    reg [7:0]  prev;
    // S2: differences
    reg        s2_v, s2_first, s2_last;
    reg signed [8:0] s2_dx, s2_dy;
    // S3: squares
    reg        s3_v, s3_first, s3_last;
    reg [16:0] s3_xx, s3_yy;
    // S4: sum
    reg        s4_v, s4_first, s4_last;
    reg [17:0] s4_e;

    always @(posedge clk) begin
        if (in_valid) begin
            s1_up <= lb[col[AB-1:0]];
            lb[col[AB-1:0]] <= in_p;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            col <= 0; row <= 0; prev <= 0;
            s1_v <= 0; s2_v <= 0; s3_v <= 0; s4_v <= 0;
            st_sharp <= 0; st_done <= 0;
        end else begin
            // S1
            s1_v <= in_valid;
            if (in_valid) begin
                s1_p     <= in_p;
                s1_left  <= prev;
                s1_hx    <= (col != 0);
                s1_hy    <= (row != 0);
                s1_first <= (col == 0) && (row == 0);
                s1_last  <= (col == wm1) && (row == hm1);
                prev     <= in_p;
                if (col == wm1) begin
                    col <= 0;
                    row <= (row == hm1) ? {CW{1'b0}} : row + 1'b1;
                end else
                    col <= col + 1'b1;
            end
            // S2
            s2_v <= s1_v;
            s2_first <= s1_first;
            s2_last  <= s1_last;
            s2_dx <= (s1_hx && s1_hy) ? ($signed({1'b0, s1_p}) - $signed({1'b0, s1_left})) : 9'sd0;
            s2_dy <= (s1_hx && s1_hy) ? ($signed({1'b0, s1_p}) - $signed({1'b0, s1_up}))   : 9'sd0;
            // S3
            s3_v <= s2_v;
            s3_first <= s2_first;
            s3_last  <= s2_last;
            s3_xx <= s2_dx * s2_dx;
            s3_yy <= s2_dy * s2_dy;
            // S4
            s4_v <= s3_v;
            s4_first <= s3_first;
            s4_last  <= s3_last;
            s4_e <= {1'b0, s3_xx} + {1'b0, s3_yy};
            // S5: accumulate
            st_done <= s4_v && s4_last;
            if (s4_v)
                st_sharp <= s4_first ? {{(SW-18){1'b0}}, s4_e} : st_sharp + {{(SW-18){1'b0}}, s4_e};
        end
    end
endmodule


module lk_pyramid #(
    parameter LEVELS = 5,
    parameter WMAX   = 1024,
    parameter CW     = 16
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire [CW-1:0]            cfg_w,      // input image size
    input  wire [CW-1:0]            cfg_h,
    input  wire                     in_valid,
    output wire                     in_ready,
    input  wire [7:0]               in_p0,      // both frames are downsampled at once (I0 and I1 chains)
    input  wire [7:0]               in_p1,
    // level l (1..LEVELS-1) output is in slice l; slice 0 is unused
    output wire [LEVELS-1:0]        lv_valid,
    output wire [8*LEVELS-1:0]      lv_p0,
    output wire [8*LEVELS-1:0]      lv_p1,
    // sharpness statistic of level-0 I0 (see lk_sharp)
    output wire [39:0]              st_sharp,
    output wire                     st_sharp_done
);
    assign in_ready = !rst;
    assign lv_valid[0]  = in_valid;
    assign lv_p0[7:0]   = in_p0;
    assign lv_p1[7:0]   = in_p1;

    lk_sharp #(.WMAX(WMAX), .CW(CW), .SW(40)) u_sharp (
        .clk(clk), .rst(rst), .cfg_w(cfg_w), .cfg_h(cfg_h), .in_valid(in_valid), .in_p(in_p0),
        .st_sharp(st_sharp), .st_done(st_sharp_done));

    genvar l;
    generate
        for (l = 1; l < LEVELS; l = l + 1) begin : lv
            lk_down #(.WMAX(WMAX >> (l - 1)), .CW(CW)) d0 (
                .clk(clk), .rst(rst), .cfg_w(cfg_w >> (l - 1)), .cfg_h(cfg_h >> (l - 1)),
                .in_valid(lv_valid[l-1]), .in_p(lv_p0[8*(l-1) +: 8]),
                .out_valid(lv_valid[l]), .out_p(lv_p0[8*l +: 8]));
            lk_down #(.WMAX(WMAX >> (l - 1)), .CW(CW)) d1 (
                .clk(clk), .rst(rst), .cfg_w(cfg_w >> (l - 1)), .cfg_h(cfg_h >> (l - 1)),
                .in_valid(lv_valid[l-1]), .in_p(lv_p1[8*(l-1) +: 8]),
                .out_valid(), .out_p(lv_p1[8*l +: 8]));
        end
    endgenerate
endmodule
