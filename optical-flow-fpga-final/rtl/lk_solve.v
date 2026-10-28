// =============================================================================
// lk_solve.v  --  Lucas-Kanade optical flow, Stage 3: 2x2 system solver (fully pipelined)
//
// Solves   [Sxx Sxy] [u]  = -2 [Sxt]        (the 2 on the right is because Stage 1 gradients are not divided by 2)
//      [Sxy Syy] [v]       [Syt]
//
//   det = Sxx*Syy - Sxy²
//   Nu  = Syy*Sxt - Sxy*Syt        u = -2·Nu/det
//   Nv  = Sxx*Syt - Sxy*Sxt        v = -2·Nv/det
//
// Outputs are Q8 fixed point (16-bit signed, pixels = out_u / 256), range +-127.996 px,
// saturating to +-32767. Pixels with det <= DET_MIN lack texture and give unreliable solutions: output 0 with out_flag = 0.
// Bit-exact with solve_fixed in python/lk_golden.py.
//
// Division is the most expensive operation in hardware. This uses a pipelined restoring divider (long division by hand):
// each stage computes one quotient bit, 15 stages in a row, so a new pixel can enter and a result leave every cycle.
// The cost is latency (about 20 cycles), but throughput stays at 1 pixel/cycle, matching the first two stages.
//
// Pipeline (v2, 250 MHz target, at most one carry chain or one multiply per stage):
//   P1 register inputs -> P2a partial products -> P2b add partial products -> P3 det/Nu/Nv -> P4a absolute values, threshold
//   -> P4b saturation check -> 16-stage divider -> output registers      total latency 23 cycles
// =============================================================================
`timescale 1ns/1ps

module lk_solve #(
    parameter CW   = 16,
    parameter SW   = 24,
    parameter [63:0] DET_MIN = 64'd65536,
    parameter FRAC = 8,                 // fractional output bits (change together with the Python model)
    parameter QB   = 15                 // quotient bits (output = 1 sign bit + QB bits)
)(
    input  wire                 clk,
    input  wire                 rst,

    input  wire                 in_valid,
    input  wire [CW-1:0]        in_x,
    input  wire [CW-1:0]        in_y,
    input  wire signed [SW-1:0] in_sxx,
    input  wire signed [SW-1:0] in_sxy,
    input  wire signed [SW-1:0] in_syy,
    input  wire signed [SW-1:0] in_sxt,
    input  wire signed [SW-1:0] in_syt,

    output reg                  out_valid,
    output reg  [CW-1:0]        out_x,
    output reg  [CW-1:0]        out_y,
    output reg  signed [QB:0]   out_u,      // Q8 fixed point
    output reg  signed [QB:0]   out_v,
    output reg                  out_flag    // 1 = valid solution
);

    localparam [QB:0] QMAX = {1'b0, {QB{1'b1}}};   // 32767
    // Divider width: dividend |N|*512 < 2^52, divisor det < 2^43, divisor << 14 < 2^57 -> 58 bits are enough
    localparam DB = 58;

    // ---------------- P1: register inputs ----------------
    reg                 v1;
    reg [CW-1:0]        x1, y1;
    // Window sums satisfy |S| <= 25 x 255 x 255 = 1,625,625 < 2^21, so 22-bit signed is enough.
    reg signed [21:0]   sxx1, sxy1, syy1, sxt1, syt1;

    // ---------------- P2a/P2b: six multiplies (two internal pipeline stages in lk_smul22) ----------------
    reg                 v2a, v2;
    reg [CW-1:0]        x2a, y2a, x2, y2;
    wire signed [43:0]  pa, pb, pc, pd, pe, pf;
    lk_smul22 u_ma (.clk(clk), .a(sxx1), .b(syy1), .p(pa));
    lk_smul22 u_mb (.clk(clk), .a(sxy1), .b(sxy1), .p(pb));
    lk_smul22 u_mc (.clk(clk), .a(syy1), .b(sxt1), .p(pc));
    lk_smul22 u_md (.clk(clk), .a(sxy1), .b(syt1), .p(pd));
    lk_smul22 u_me (.clk(clk), .a(sxx1), .b(syt1), .p(pe));
    lk_smul22 u_mf (.clk(clk), .a(sxy1), .b(sxt1), .p(pf));

    // ---------------- P3: det, numerators ----------------
    reg                 v3;
    reg [CW-1:0]        x3, y3;
    reg signed [48:0]   det3, nu3, nv3;

    // ---------------- P4a: absolute values, threshold check ----------------
    reg                 v4a;
    reg [CW-1:0]        x4a, y4a;
    reg                 flag4a, sgnu4a, sgnv4a;
    reg [48:0]          magu4a, magv4a, det4a;

    // ---------------- P4b: saturation check, feed the divider ----------------
    reg                 v4;
    reg [CW-1:0]        x4, y4;
    reg                 flag4, sgnu4, sgnv4, satu4, satv4;
    reg [DB-1:0]        dvdu4, dvdv4, dvs4;

    wire [48:0] magu = nu3[48] ? (~nu3 + 1'b1) : nu3;    // |Nu|
    wire [48:0] magv = nv3[48] ? (~nv3 + 1'b1) : nv3;
    wire        flag = !det3[48] && (det3 > $signed({1'b0, DET_MIN}));
    wire [63:0] dvdu = {15'b0, magu4a} << (FRAC + 1);    // |Nu| * 2 * 256
    wire [63:0] dvdv = {15'b0, magv4a} << (FRAC + 1);
    wire [63:0] dvs  = {15'b0, det4a};
    wire [63:0] lim  = dvs << QB;                         // saturate when quotient >= 2^QB

    always @(posedge clk) begin
        if (rst) begin
            v1 <= 1'b0; v2a <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4a <= 1'b0; v4 <= 1'b0;
        end else begin
            v1 <= in_valid; v2a <= v1; v2 <= v2a; v3 <= v2; v4a <= v3; v4 <= v4a;
        end
        // P1
        x1 <= in_x;  y1 <= in_y;
        sxx1 <= in_sxx[21:0];  sxy1 <= in_sxy[21:0];  syy1 <= in_syy[21:0];
        sxt1 <= in_sxt[21:0];  syt1 <= in_syt[21:0];
        // P2a, P2b (multiplies inside lk_smul22)
        x2a <= x1;  y2a <= y1;  x2 <= x2a;  y2 <= y2a;
        // P3
        x3 <= x2;  y3 <= y2;
        det3 <= pa - pb;
        nu3  <= pc - pd;
        nv3  <= pe - pf;
        // P4a
        x4a <= x3;  y4a <= y3;
        flag4a <= flag;
        sgnu4a <= nu3[48];  sgnv4a <= nv3[48];
        magu4a <= magu;     magv4a <= magv;     det4a <= det3;
        // P4b
        x4 <= x4a;  y4 <= y4a;
        flag4 <= flag4a;  sgnu4 <= sgnu4a;  sgnv4 <= sgnv4a;
        satu4 <= (dvdu >= lim);
        satv4 <= (dvdv >= lim);
        dvdu4 <= dvdu[DB-1:0];  dvdv4 <= dvdv[DB-1:0];  dvs4 <= dvs[DB-1:0];
    end

    // ---------------- divider (latency QB+1 cycles) ----------------
    wire [QB-1:0] qu, qv;
    lk_divpipe #(.NB(DB), .QB(QB)) u_divu (.clk(clk), .dividend(dvdu4), .divisor(dvs4), .quotient(qu));
    lk_divpipe #(.NB(DB), .QB(QB)) u_divv (.clk(clk), .dividend(dvdv4), .divisor(dvs4), .quotient(qv));

    // side-band signals (valid, coordinates, sign, saturation) are delayed QB+1 cycles along with the divider
    localparam SBW = 1 + 2*CW + 5;
    reg  [SBW*(QB+1)-1:0] sb;
    wire [SBW-1:0] sb_in  = {v4, x4, y4, flag4, sgnu4, sgnv4, satu4, satv4};
    wire [SBW-1:0] sb_out = sb[SBW*QB +: SBW];

    always @(posedge clk) begin
        if (rst) sb <= {SBW*(QB+1){1'b0}};
        else     sb <= {sb[SBW*QB-1:0], sb_in};
    end

    wire          o_v    = sb_out[SBW-1];
    wire [CW-1:0] o_x    = sb_out[SBW-2 -: CW];
    wire [CW-1:0] o_y    = sb_out[SBW-2-CW -: CW];
    wire          o_flag = sb_out[4];
    wire          o_sgnu = sb_out[3];
    wire          o_sgnv = sb_out[2];
    wire          o_satu = sb_out[1];
    wire          o_satv = sb_out[0];

    wire [QB:0] mu = o_satu ? QMAX : {1'b0, qu};
    wire [QB:0] mv = o_satv ? QMAX : {1'b0, qv};

    // ---------------- output registers ----------------
    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0; out_x <= 0; out_y <= 0;
            out_u <= 0; out_v <= 0; out_flag <= 1'b0;
        end else begin
            out_valid <= o_v;
            out_x     <= o_x;
            out_y     <= o_y;
            out_flag  <= o_flag;
            // u = -2N/det: u is positive when N is negative
            out_u <= !o_flag ? {(QB+1){1'b0}} : (o_sgnu ? mu : (~mu + 1'b1));
            out_v <= !o_flag ? {(QB+1){1'b0}} : (o_sgnv ? mv : (~mv + 1'b1));
        end
    end

endmodule


// =============================================================================
// lk_divpipe -- pipelined unsigned divider (restoring division)
//   Requires dividend < (divisor << QB) (the caller checks saturation first); outputs floor(dividend / divisor).
//   Stage 0 registers the inputs; stage s (1..QB) decides quotient bit QB-s:
//       if remainder >= divisor << (QB-s): remainder -= divisor << (QB-s), and that quotient bit = 1
//   Latency QB+1 cycles, one new operand per cycle.
// =============================================================================
module lk_divpipe #(
    parameter NB = 64,
    parameter QB = 15
)(
    input  wire          clk,
    input  wire [NB-1:0] dividend,
    input  wire [NB-1:0] divisor,
    output wire [QB-1:0] quotient
);
    reg [NB*(QB+1)-1:0] R;      // remainder of each stage, packed into one vector
    reg [NB*(QB+1)-1:0] V;      // divisor of each stage
    reg [QB*(QB+1)-1:0] Q;      // quotient bits computed so far at each stage

    always @(posedge clk) begin
        R[0 +: NB] <= dividend;
        V[0 +: NB] <= divisor;
        Q[0 +: QB] <= {QB{1'b0}};
    end

    genvar s;
    generate
        for (s = 1; s <= QB; s = s + 1) begin : st
            wire [NB-1:0] r_prev = R[(s-1)*NB +: NB];
            wire [NB-1:0] v_prev = V[(s-1)*NB +: NB];
            wire [QB-1:0] q_prev = Q[(s-1)*QB +: QB];
            wire [NB-1:0] v_sh   = v_prev << (QB - s);
            wire [QB-1:0] bit_s  = {{(QB-1){1'b0}}, 1'b1} << (QB - s);
            // a single subtractor: compute one extra bit, and its top (borrow) bit is the "too small" flag,
            // so compare and subtract share one carry chain -- half the LUTs of writing >= and - separately
            wire [NB:0]   diff   = {1'b0, r_prev} - {1'b0, v_sh};
            wire          ge     = !diff[NB];
            always @(posedge clk) begin
                V[s*NB +: NB] <= v_prev;
                R[s*NB +: NB] <= ge ? diff[NB-1:0] : r_prev;
                Q[s*QB +: QB] <= ge ? (q_prev | bit_s) : q_prev;
            end
        end
    endgenerate

    assign quotient = Q[QB*QB +: QB];
endmodule


// =============================================================================
// lk_smul22 -- 22-bit x 22-bit signed multiply, split into two partial products
//   The Xilinx DSP48E1 multiplier is 25 x 18 bits. A 22-bit b does not fit the 18-bit port,
//   so b is split into: high 5 bits (signed) x 2^17 + low 17 bits (unsigned),
//       a*b = (a * b_hi) << 17 + a * b_lo
//   each partial product uses 1 DSP, 2 in total. v2 adds a register stage after the partial products (250 MHz).
// =============================================================================
module lk_smul22 (
    input  wire               clk,
    input  wire signed [21:0] a,
    input  wire signed [21:0] b,
    output reg  signed [43:0] p             // latency 2 cycles
);
    wire signed [17:0] b_lo = {1'b0, b[16:0]};     // low 17 bits, zero-extended as positive
    wire signed [4:0]  b_hi = b[21:17];             // high 5 bits, keeps the sign
    reg  signed [39:0] p_lo;                        // 22 x 18
    reg  signed [26:0] p_hi;                        // 22 x 5
    always @(posedge clk) begin
        p_lo <= a * b_lo;                           // cycle 1: two partial products (DSP internal registers)
        p_hi <= a * b_hi;
        p    <= $signed({p_hi, 17'b0}) + p_lo;      // cycle 2: add
    end
endmodule
