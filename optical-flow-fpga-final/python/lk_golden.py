"""
Lucas-Kanade optical flow -- software golden model

Purpose:
  1. A complete integer implementation of Lucas-Kanade that does exactly what the hardware does (bit-exact),
     so the Verilog simulation output can be compared with it pixel by pixel.
  2. Generates synthetic test images (with known true displacement) to show the algorithm itself gets it right.
  3. Exports .hex test vectors for the Verilog testbenches ($readmemh).

The three hardware pipeline stages (all implemented here and in rtl/):
  Stage 1  gradient unit    : Ix, Iy, It
  Stage 2  window accumulator: over a 5x5 window, sum Ix², IxIy, Iy², IxIt, IyIt
  Stage 3  solver           : solve the 2x2 linear system -> (u, v)

Conventions (hardware and software must agree):
  - 8-bit grayscale images, 0..255
  - Ix(y,x) = I0(y, x+1) - I0(y, x-1)     central difference, not divided by 2 (true gradient = Ix/2)
  - Iy(y,x) = I0(y+1, x) - I0(y-1, x)
  - It(y,x) = I1(y, x)   - I0(y, x)
  - results in [-255, 255], stored as 9-bit signed (two's complement)
  - the outermost ring of pixels (x=0, x=W-1, y=0, y=H-1) outputs 0 for all three
"""
import argparse
import os
import numpy as np


# ---------------------------------------------------------------- test images
def make_frames(W, H, dx, dy, seed=0):
    """Synthesise an image pair: frame1 is frame0 shifted by (dx, dy) pixels (may be fractional).
    The texture is a sum of sines, so fractional shifts can be computed analytically without interpolation."""
    rng = np.random.default_rng(seed)
    comps = []
    for _ in range(6):
        kx, ky = rng.uniform(-0.35, 0.35, size=2)   # spatial frequency (rad/pixel); not too high or LK breaks down
        ph = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(10, 25)
        comps.append((kx, ky, ph, amp))

    def render(shift_x, shift_y):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        img = np.full((H, W), 128.0)
        for kx, ky, ph, amp in comps:
            img += amp * np.sin(kx * (xx - shift_x) + ky * (yy - shift_y) + ph)
        return np.clip(np.round(img), 0, 255).astype(np.int32)

    return render(0.0, 0.0), render(dx, dy)


# ---------------------------------------------------------------- Stage 1: gradients
def gradients(I0, I1):
    H, W = I0.shape
    Ix = np.zeros((H, W), np.int32)
    Iy = np.zeros((H, W), np.int32)
    It = np.zeros((H, W), np.int32)
    Ix[1:-1, 1:-1] = I0[1:-1, 2:] - I0[1:-1, :-2]
    Iy[1:-1, 1:-1] = I0[2:, 1:-1] - I0[:-2, 1:-1]
    It[1:-1, 1:-1] = I1[1:-1, 1:-1] - I0[1:-1, 1:-1]
    return Ix, Iy, It


# ---------------------------------------------------------------- Stage 2: window sums
def window_sums(Ix, Iy, It, win=5):
    """Sums of the 5 products over the win x win window around each pixel (integers; an adder tree / sliding window in hardware)."""
    r = win // 2
    prods = {
        "Sxx": Ix * Ix, "Sxy": Ix * Iy, "Syy": Iy * Iy,
        "Sxt": Ix * It, "Syt": Iy * It,
    }
    H, W = Ix.shape
    out = {}
    for k, p in prods.items():
        p = p.astype(np.int64)
        s = np.zeros((H, W), np.int64)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                s[r:H - r, r:W - r] += p[r + dy:H - r + dy, r + dx:W - r + dx]
        out[k] = s
    return out


# ---------------------------------------------------------------- Stage 3: solve
def solve_flow(S, min_det=1.0):
    """Solve  [Sxx Sxy][u]   = -2 [Sxt]
          [Sxy Syy][v]        [Syt]
    The 2 on the right comes from the "gradients are not divided by 2" convention (true gradient = Ix/2).
    A tiny det means too little texture (the aperture problem); such pixels cannot be solved reliably and are marked invalid."""
    Sxx, Sxy, Syy, Sxt, Syt = (S[k].astype(np.float64) for k in ("Sxx", "Sxy", "Syy", "Sxt", "Syt"))
    det = Sxx * Syy - Sxy * Sxy
    valid = det > min_det
    safe = np.where(valid, det, 1.0)
    u = -2.0 * (Syy * Sxt - Sxy * Syt) / safe
    v = -2.0 * (Sxx * Syt - Sxy * Sxt) / safe
    return np.where(valid, u, 0.0), np.where(valid, v, 0.0), valid


# ---------------------------------------------------------------- Stage 3 (fixed point): bit-exact with the hardware
FRAC = 8                    # fractional bits of the output flow: u_q = u * 256
QMAX = (1 << 15) - 1        # largest 16-bit signed output (saturation)
DET_MIN = 1 << 16           # det at or below this means "too little texture": output 0, flag=0


def solve_fixed(S, det_min=DET_MIN):
    """Exact model of the hardware lk_solve.v, all integer arithmetic.
        det = Sxx*Syy - Sxy^2                  (>= 0 by Cauchy-Schwarz)
        Nu  = Syy*Sxt - Sxy*Syt,  Nv = Sxx*Syt - Sxy*Sxt
        u   = -2 * Nu / det   ->  fixed point: u_q = -sign(Nu) * min(floor(|Nu| * 2 * 256 / det), 32767)
    Returns (u_q, v_q, flag); u_q/v_q are Q8 fixed-point integers (divide by 256 for pixels)."""
    Sxx, Sxy, Syy, Sxt, Syt = (S[k].astype(np.int64) for k in ("Sxx", "Sxy", "Syy", "Sxt", "Syt"))
    det = Sxx * Syy - Sxy * Sxy
    nu = Syy * Sxt - Sxy * Syt
    nv = Sxx * Syt - Sxy * Sxt
    flag = det > det_min
    safe = np.where(flag, det, 1)

    def q(n):
        mag = np.minimum((np.abs(n) << (FRAC + 1)) // safe, QMAX)
        return np.where(n < 0, mag, -mag)          # u = -2N/det, so the sign is flipped
    u_q = np.where(flag, q(nu), 0)
    v_q = np.where(flag, q(nv), 0)
    return u_q.astype(np.int64), v_q.astype(np.int64), flag


def lk_fixed(I0, I1, det_min=DET_MIN):
    """Software equivalent of the whole hardware pipeline: 8-bit grayscale -> (u_q, v_q, flag) plus intermediate results."""
    Ix, Iy, It = gradients(I0.astype(np.int32), I1.astype(np.int32))
    S = window_sums(Ix, Iy, It)
    u_q, v_q, flag = solve_fixed(S, det_min)
    return (Ix, Iy, It), S, (u_q, v_q, flag)


# ---------------------------------------------------------------- export test vectors
_HEX_TABLES = {}


def write_hex(path, arr, bits):
    """One hex number per line (two's complement) for Verilog $readmemh. Vectorised: a million lines take a second or two."""
    mask = (1 << bits) - 1
    digits = (bits + 3) // 4
    vals = np.asarray(arr).reshape(-1).astype(np.int64) & mask
    if digits <= 4:
        if digits not in _HEX_TABLES:
            _HEX_TABLES[digits] = np.array([f"{i:0{digits}x}" for i in range(16 ** digits)])
        strs = _HEX_TABLES[digits][vals]
    else:
        strs = np.char.mod(f"%0{digits}x", vals)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(strs.tolist()))
        f.write("\n")


def write_multi_vectors(out, frame_pairs, det_min=DET_MIN):
    """For the multi-lane testbench (tb_lk_multi.v): LANES frame pairs concatenated lane by lane into one file.
    Returns (u_q, v_q, flag) for each lane."""
    H, W = frame_pairs[0][0].shape
    res = [lk_fixed(I0, I1, det_min)[2] for I0, I1 in frame_pairs]
    os.makedirs(out, exist_ok=True)
    write_hex(os.path.join(out, "frame0.hex"), np.concatenate([p[0].reshape(-1) for p in frame_pairs]), 8)
    write_hex(os.path.join(out, "frame1.hex"), np.concatenate([p[1].reshape(-1) for p in frame_pairs]), 8)
    write_hex(os.path.join(out, "golden_u.hex"), np.concatenate([r[0].reshape(-1) for r in res]), 16)
    write_hex(os.path.join(out, "golden_v.hex"), np.concatenate([r[1].reshape(-1) for r in res]), 16)
    write_hex(os.path.join(out, "golden_flag.hex"), np.concatenate([r[2].reshape(-1) for r in res]).astype(np.int64), 1)
    write_params(out, W, H, det_min, lanes=len(frame_pairs))
    return res


def write_params(out, W, H, det_min, lanes=1):
    with open(os.path.join(out, "params.vh"), "w", encoding="utf-8", newline="\n") as f:
        f.write("// Generated together with the test vectors. Do not edit; rerun lk_golden.py\n")
        f.write(f"localparam IMG_W = {W};\nlocalparam IMG_H = {H};\n")
        f.write(f"localparam [63:0] DET_MIN = 64'd{det_min};\n")
        f.write(f"localparam LANES = {lanes};\n")
        # line buffers are sized by WMAX and the width is given at run time; WMAX is deliberately a power of 2 >= W so WMAX != W is tested too
        f.write(f"localparam WMAX = {wmax_for(W)};\n")


def wmax_for(W):
    return max(16, 1 << (int(W) - 1).bit_length())


def stress_frames(W, H):
    """Bottom half: 0/255 checkerboard with the second frame inverted -> gradients, S and det use the full bit width
    Top half: almost flat image (100/101) + random second frame -> tiny det, large numerators -> exercises saturation and the threshold"""
    rng = np.random.default_rng(150)
    yy, xx = np.mgrid[0:H, 0:W]
    I0 = np.where(((xx // 2 + yy // 2) % 2) == 0, 0, 255).astype(np.int64)
    I1 = 255 - I0
    I0[:H // 2] = 100 + rng.integers(0, 2, (H // 2, W))
    I1[:H // 2] = rng.integers(0, 256, (H // 2, W))
    return I0, I1


def noise_frames(W, H, seed=0):
    rng = np.random.default_rng(seed)
    I0 = rng.integers(0, 256, size=(H, W)).astype(np.int32)
    I1 = rng.integers(0, 256, size=(H, W)).astype(np.int32)
    I0[2, 1:4] = [0, 128, 255]      # a few extreme values placed by hand so that Ix hits exactly +255 and -255
    I0[3, 1:4] = [255, 128, 0]
    I0[7, 10], I0[9, 10] = 0, 255   # Iy(8,10) = +255
    I0[7, 12], I0[9, 12] = 255, 0   # Iy(8,12) = -255
    I1[5, 5], I0[5, 5] = 255, 0     # It = +255
    I1[6, 6], I0[6, 6] = 0, 255     # It = -255
    return I0, I1


def write_all_vectors(out, I0, I1, det_min=DET_MIN):
    """Write all files the testbenches need (golden answers for every pipeline stage)."""
    (Ix, Iy, It), S, (u_q, v_q, flag) = lk_fixed(I0, I1, det_min)
    H, W = I0.shape
    os.makedirs(out, exist_ok=True)
    write_hex(os.path.join(out, "frame0.hex"), I0, 8)
    write_hex(os.path.join(out, "frame1.hex"), I1, 8)
    write_hex(os.path.join(out, "golden_ix.hex"), Ix, 9)
    write_hex(os.path.join(out, "golden_iy.hex"), Iy, 9)
    write_hex(os.path.join(out, "golden_it.hex"), It, 9)
    for k in ("Sxx", "Sxy", "Syy", "Sxt", "Syt"):
        write_hex(os.path.join(out, f"golden_{k.lower()}.hex"), S[k], 24)
    write_hex(os.path.join(out, "golden_u.hex"), u_q, 16)
    write_hex(os.path.join(out, "golden_v.hex"), v_q, 16)
    write_hex(os.path.join(out, "golden_flag.hex"), flag.astype(np.int64), 1)
    write_params(out, W, H, det_min)
    return (Ix, Iy, It), S, (u_q, v_q, flag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--dx", type=float, default=0.6)
    ap.add_argument("--dy", type=float, default=-0.4)
    ap.add_argument("--noise", action="store_true",
                    help="use random-noise images instead (the flow is meaningless, but gradients cover the full -255..255 range to test sign bits/widths)")
    ap.add_argument("--stress", action="store_true",
                    help="solver edge cases: positive/negative saturation, det just below the threshold, S/det at full width")
    ap.add_argument("--det-min", type=int, default=None, help=f"det threshold, default {DET_MIN}")
    ap.add_argument("--lanes", type=int, default=0,
                    help="generate multi-lane test vectors: one different image pair per lane (smooth/noise/stress in turn)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "vectors"))
    a = ap.parse_args()

    det_min = DET_MIN if a.det_min is None else a.det_min
    if a.lanes:
        pairs = []
        for k in range(a.lanes):
            if k % 3 == 1:
                pairs.append(noise_frames(a.W, a.H, seed=k))
            elif k % 3 == 2:
                pairs.append(stress_frames(a.W, a.H))
            else:
                pairs.append(make_frames(a.W, a.H, 0.3 + 0.2 * k, -0.2 * k, seed=k))
        dm = 256 if a.det_min is None else a.det_min
        write_multi_vectors(a.out, pairs, dm)
        print(f"[multi-lane] {a.lanes} image pairs {a.W}x{a.H}, det_min={dm}, vectors written to {os.path.abspath(a.out)}")
        return
    if a.stress:
        I0, I1 = stress_frames(a.W, a.H)
        if a.det_min is None:
            det_min = 256
    elif a.noise:
        I0, I1 = noise_frames(a.W, a.H, a.seed)
    else:
        I0, I1 = make_frames(a.W, a.H, a.dx, a.dy, a.seed)
    os.makedirs(a.out, exist_ok=True)
    (Ix, Iy, It), S, (u_q, v_q, flag) = write_all_vectors(a.out, I0, I1, det_min)
    u, v, valid = solve_flow(S)

    # only pixels far enough from the border with a valid solution
    m = valid.copy()
    m[:3, :] = m[-3:, :] = False
    m[:, :3] = m[:, -3:] = False
    if a.noise:
        print(f"[noise test] image {a.W}x{a.H} (flow is meaningless; only tests widths/signs)")
    print(f"image {a.W}x{a.H}, true displacement (dx, dy) = ({a.dx}, {a.dy})")
    print(f"LK median estimate (u, v) = ({np.median(u[m]):.3f}, {np.median(v[m]):.3f})  "
          f"[valid pixels {m.sum()}/{m.size}]")
    print(f"gradient range: Ix [{Ix.min()}, {Ix.max()}]  Iy [{Iy.min()}, {Iy.max()}]  It [{It.min()}, {It.max()}]")

    if a.stress:
        print(f"[stress test] positive saturation {int((u_q == QMAX).sum() + (v_q == QMAX).sum())}, "
              f"negative saturation {int((u_q == -QMAX).sum() + (v_q == -QMAX).sum())}, det_min = {det_min}")
    fm = flag & m
    print(f"fixed point (hardware) median (u, v) = ({np.median(u_q[fm]) / 256:.3f}, {np.median(v_q[fm]) / 256:.3f})  "
          f"[flag=1 pixels {fm.sum()}]")
    print(f"test vectors written to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
