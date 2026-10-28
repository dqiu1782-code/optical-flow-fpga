"""
lk_pyramid.py -- software golden model of pyramid (multi-scale) Lucas-Kanade, bit-exact with the hardware

Why a pyramid: single-scale LK only works for motion below ~1.5 px (the Taylor expansion only holds locally).
Shrinking the image by 2^k shrinks the motion by 2^k too. So the coarse motion is estimated on the smallest image first
and refined level by level: each level first shifts the second frame back by the previous estimate (warp) and only solves the small residual.

All integer arithmetic (the hardware matches it bit for bit):

  1. Downsample   I_{l+1}(y,x) = (a + b + c + d + 2) >> 2,  a..d = the 2x2 block of I_l; size W>>1, H>>1 (rounded down)
  2. Initial guess g(y,x) = 2 * flow_{l+1}(min(y>>1, Hc-1), min(x>>1, Wc-1))   (Q8)
              gu clamped to +-QMAX; gv clamped to +-RY*256 (vertically limited by the number of buffered rows, horizontally unlimited)
              g = 0 at the coarsest level
  3. Bilinear warp (Q8 fraction):
        px = clip((x<<8) + gu, 0, ((W-1)<<8) - 1),  xi = px>>8, fx = px & 255   (same for y)
        top = I1[yi, xi]*(256-fx) + I1[yi, xi+1]*fx
        bot = I1[yi+1, xi]*(256-fx) + I1[yi+1, xi+1]*fx
        I1w(y,x) = (top*(256-fy) + bot*fy + 32768) >> 16
  4. LK at this level:  (du, dv, flag) = lk_fixed(I0_l, I1w)    (the original 3-stage pipeline)
  5. Merge        u = clip(gu + du, +-QMAX),  v = clip(gv + dv, +-QMAX)
              (pixels with flag=0 have du=dv=0, i.e. keep the guess)

  The finest level (full resolution) does not write the dense flow map; it only outputs per-frame statistics (over flag=1 pixels):
     count, sum_u, sum_v, sum_l1 (|u|+|v|), max_l1, 8-bin histogram (|u|+|v| binned at
     0.5/1/2/4/8/16/32 px)
"""
import os
import numpy as np

import lk_golden as G

LEVELS = 5           # 1024x436 -> 512x218 -> 256x109 -> 128x54 -> 64x27
NRB = 6              # the hardware warp's ring line buffer has 2^NRB rows (64 rows, one URAM per bank)
RY = (1 << NRB) // 2 - 2   # = 30: maximum vertical initial guess (pixels); the buffer holds 2*RY+4 rows
DU_MAX = 256         # default per-level residual clamp +-1 px (Q8); on Sintel EPE 12.19 -> 11.26
QMAX = G.QMAX
HIST_EDGES = (128, 256, 512, 1024, 2048, 4096, 8192)      # Q8: 0.5, 1, 2, 4, 8, 16, 32 px


# ---------------------------------------------------------------- 1. pyramid
def downsample(I):
    H, W = I.shape
    h, w = H >> 1, W >> 1
    a = I[0:2 * h:2, 0:2 * w:2].astype(np.int32)
    b = I[0:2 * h:2, 1:2 * w:2]
    c = I[1:2 * h:2, 0:2 * w:2]
    d = I[1:2 * h:2, 1:2 * w:2]
    return ((a + b + c + d + 2) >> 2).astype(np.int32)


def build_pyramid(I, levels=LEVELS):
    pyr = [np.asarray(I, dtype=np.int32)]
    for _ in range(levels - 1):
        pyr.append(downsample(pyr[-1]))
    return pyr


def level_dims(W, H, levels=LEVELS):
    return [(W >> l, H >> l) for l in range(levels)]


# ---------------------------------------------------------------- 2. initial guess
def upsample_guess(uc, vc, W, H, ry=RY):
    Hc, Wc = uc.shape
    ys = np.minimum(np.arange(H) >> 1, Hc - 1)
    xs = np.minimum(np.arange(W) >> 1, Wc - 1)
    gu = np.clip(2 * uc[np.ix_(ys, xs)], -QMAX, QMAX)
    gv = np.clip(2 * vc[np.ix_(ys, xs)], -ry * 256, ry * 256)
    return gu.astype(np.int64), gv.astype(np.int64)


# ---------------------------------------------------------------- 3. warp
def warp_bilinear(I1, gu, gv):
    H, W = I1.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.int64)
    px = np.clip((xx << 8) + gu, 0, ((W - 1) << 8) - 1)
    py = np.clip((yy << 8) + gv, 0, ((H - 1) << 8) - 1)
    xi, fx = px >> 8, px & 255
    yi, fy = py >> 8, py & 255
    I = I1.astype(np.int64)
    top = I[yi, xi] * (256 - fx) + I[yi, xi + 1] * fx
    bot = I[yi + 1, xi] * (256 - fx) + I[yi + 1, xi + 1] * fx
    return ((top * (256 - fy) + bot * fy + 32768) >> 16).astype(np.int32)


# ---------------------------------------------------------------- 4+5. the whole pyramid
def lk_pyramid_fixed(I0, I1, levels=LEVELS, ry=RY, det_min=G.DET_MIN, keep=False, du_max=0, det_mul_coarse=1):
    """Returns (u, v, flag) at full resolution, Q8; with keep=True also returns every level's intermediate results (for the testbench)."""
    P0, P1 = build_pyramid(I0, levels), build_pyramid(I1, levels)
    trace = []
    u = v = None
    for l in range(levels - 1, -1, -1):
        H, W = P0[l].shape
        if u is None:
            gu = np.zeros((H, W), np.int64)
            gv = np.zeros((H, W), np.int64)
        else:
            gu, gv = upsample_guess(u, v, W, H, ry)
        I1w = warp_bilinear(P1[l], gu, gv)
        _, _, (du, dv, flag) = G.lk_fixed(P0[l], I1w, det_min * (det_mul_coarse if l > 0 else 1))
        dur, dvr = du, dv
        if du_max:                       # (optional) per-level residual clamp, so a bad solution at one level is not amplified into the next
            dur = np.clip(du, -du_max, du_max)
            dvr = np.clip(dv, -du_max, du_max)
        u = np.clip(gu + dur, -QMAX, QMAX)
        v = np.clip(gv + dvr, -QMAX, QMAX)
        if keep:
            trace.append({"level": l, "W": W, "H": H, "I0": P0[l], "I1": P1[l], "gu": gu, "gv": gv,
                          "I1w": I1w, "du": du, "dv": dv, "u": u, "v": v, "flag": flag})
    if keep:
        return u, v, flag, trace
    return u, v, flag


# ---------------------------------------------------------------- per-frame statistics (model of the on-chip statistics unit)
def frame_stats(u, v, flag):
    m = flag.astype(bool)
    uu, vv = u[m].astype(np.int64), v[m].astype(np.int64)
    l1 = np.abs(uu) + np.abs(vv)
    hist = np.bincount(np.searchsorted(np.array(HIST_EDGES), l1, side="right"), minlength=8)
    return {"count": int(m.sum()), "sum_u": int(uu.sum()), "sum_v": int(vv.sum()),
            "sum_l1": int(l1.sum()), "max_l1": int(l1.max()) if l1.size else 0,
            "hist": [int(h) for h in hist]}


def frame_sharp(I):
    """Sharpness statistic (model of RTL lk_sharp): sum over y>=1, x>=1 of (I[y,x]-I[y,x-1])² + (I[y,x]-I[y-1,x])²"""
    I = I.astype(np.int64)
    dx = I[1:, 1:] - I[1:, :-1]
    dy = I[1:, 1:] - I[:-1, 1:]
    return int((dx * dx).sum() + (dy * dy).sum())


def stats_summary(st):
    """Convert the integer statistics in the hardware registers to pixel units (what software does after reading them)."""
    n = max(st["count"], 1)
    hist = np.array(st["hist"])
    # median bin estimated from the histogram
    cum = np.cumsum(hist)
    med_bin = int(np.searchsorted(cum, cum[-1] / 2)) if cum[-1] else 0
    edges_px = [0] + [e / 256 for e in HIST_EDGES] + [np.inf]
    return {"mean_u_px": st["sum_u"] / n / 256, "mean_v_px": st["sum_v"] / n / 256,
            "mean_l1_px": st["sum_l1"] / n / 256, "max_l1_px": st["max_l1"] / 256,
            "median_bin_px": (edges_px[med_bin], edges_px[med_bin + 1]), "valid_frac": None}


# ---------------------------------------------------------------- synthetic large-motion test images
def shifted_pair(W, H, dx, dy, seed=0):
    """Richer texture (multi-scale sines + smooth noise), shifted by (dx, dy). Used to test large displacements."""
    rng = np.random.default_rng(seed)
    comps = []
    for k in range(10):
        f = rng.uniform(0.02, 0.3)
        th = rng.uniform(0, np.pi)
        comps.append((f * np.cos(th), f * np.sin(th), rng.uniform(0, 2 * np.pi), rng.uniform(6, 18)))

    def render(sx, sy):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        img = np.full((H, W), 128.0)
        for kx, ky, ph, amp in comps:
            img += amp * np.sin(kx * (xx - sx) + ky * (yy - sy) + ph)
        return np.clip(np.round(img), 0, 255).astype(np.int32)
    return render(0, 0), render(dx, dy)


def _median_flow(u, v, flag, b=4):
    m = flag.copy()
    m[:b] = m[-b:] = False
    m[:, :b] = m[:, -b:] = False
    return np.median(u[m]) / 256, np.median(v[m]) / 256, m.mean()


# ---------------------------------------------------------------- test vectors for tb_lk_pyr.v
def level_offsets(W, H, levels):
    offs, o = [], 0
    for (w, h) in level_dims(W, H, levels):
        offs.append(o)
        o += w * h
    return offs, o


def write_pyr_vectors(out, I0, I1, levels=LEVELS, nrb=NRB, det_min=G.DET_MIN, du_max=0):
    """Golden answers of all levels concatenated level by level (level l starts at offs[l]): pyramid, warp result, du/dv, final u/v/flag;
    plus the level-0 statistics registers. Returns (u, v, flag, stats)."""
    ry = (1 << nrb) // 2 - 2
    H, W = I0.shape
    u, v, flag, tr = lk_pyramid_fixed(I0, I1, levels, ry, det_min, keep=True, du_max=du_max)
    tr = sorted(tr, key=lambda t: t["level"])
    st = frame_stats(u, v, flag)
    os.makedirs(out, exist_ok=True)
    cat = lambda k: np.concatenate([t[k].reshape(-1) for t in tr])
    G.write_hex(os.path.join(out, "frame0.hex"), I0, 8)
    G.write_hex(os.path.join(out, "frame1.hex"), I1, 8)
    G.write_hex(os.path.join(out, "golden_pyr0.hex"), cat("I0"), 8)
    G.write_hex(os.path.join(out, "golden_pyr1.hex"), cat("I1"), 8)
    G.write_hex(os.path.join(out, "golden_i1w.hex"), cat("I1w"), 8)
    G.write_hex(os.path.join(out, "golden_du.hex"), cat("du"), 16)
    G.write_hex(os.path.join(out, "golden_dv.hex"), cat("dv"), 16)
    G.write_hex(os.path.join(out, "golden_fu.hex"), cat("u"), 16)
    G.write_hex(os.path.join(out, "golden_fv.hex"), cat("v"), 16)
    G.write_hex(os.path.join(out, "golden_ff.hex"), cat("flag").astype(np.int64), 1)
    G.write_hex(os.path.join(out, "golden_stats.hex"),
                [st["count"], st["sum_u"], st["sum_v"], st["sum_l1"], st["max_l1"], *st["hist"]], 40)
    offs, tot = level_offsets(W, H, levels)
    dims = level_dims(W, H, levels)
    with open(os.path.join(out, "params.vh"), "w", encoding="utf-8", newline="\n") as f:
        f.write("// Generated by lk_pyramid.py together with the test vectors\n")
        f.write(f"localparam IMG_W = {W};\nlocalparam IMG_H = {H};\n")
        f.write(f"localparam LEVELS = {levels};\nlocalparam NRB = {nrb};\n")
        f.write(f"localparam WMAX = {G.wmax_for(W)};\nlocalparam TOT = {tot};\n")
        f.write(f"localparam [63:0] DET_MIN = 64'd{det_min};\n")
        f.write(f"localparam DU_MAX = {du_max};\n")
        f.write(f"localparam [39:0] EXP_SHARP = 40'd{frame_sharp(I0)};\n")
        for name, vals in (("lvl_w", [d[0] for d in dims]), ("lvl_h", [d[1] for d in dims]), ("lvl_off", offs)):
            f.write(f"function integer {name};\n    input integer l;\n    begin\n        case (l)\n")
            for k, val in enumerate(vals):
                f.write(f"            {k}: {name} = {val};\n")
            f.write(f"            default: {name} = 0;\n        endcase\n    end\nendfunction\n")
    return u, v, flag, st


def pyr_test_frames(kind, W, H, seed=0):
    """Pyramid test images: large-motion shift / random noise (covers extremes) / large vertical motion (tests the RY clamp)"""
    if kind == "shift":
        return shifted_pair(W, H, 5.6, -3.3, seed)
    if kind == "vert":
        return shifted_pair(W, H, -2.5, 19.0, seed)        # vertical 19 px > RY=14 (with --nrb 5) triggers the gv clamp
    if kind == "stress":
        I0, I1 = G.stress_frames(W, H)                    # with det_min=256: du saturates to +-32767, and both g and u get clamped
        return I0.astype(np.int32), I1.astype(np.int32)
    if kind == "noise":
        rng = np.random.default_rng(seed)
        I0 = rng.integers(0, 256, (H, W)).astype(np.int32)
        I1 = np.roll(I0, (2, -3), axis=(0, 1))
        I1 = np.clip(I1 + rng.integers(-20, 21, (H, W)), 0, 255).astype(np.int32)
        return I0, I1
    raise ValueError(kind)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--vectors", default="", help="write test vectors for tb_lk_pyr.v to this directory")
    ap.add_argument("--kind", default="shift", choices=["shift", "vert", "noise", "stress"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--det-min", type=int, default=G.DET_MIN)
    ap.add_argument("--du-max", type=int, default=0, help="per-level residual clamp (Q8), 0 = off")
    ap.add_argument("--nrb", type=int, default=NRB, help="warp ring buffer has 2^NRB rows (RY = 2^NRB/2 - 2)")
    a = ap.parse_args()
    if a.vectors:
        I0, I1 = pyr_test_frames(a.kind, a.W, a.H, a.seed)
        u, v, f, st = write_pyr_vectors(a.vectors, I0, I1, a.levels, a.nrb, a.det_min, a.du_max)
        mu, mv, fr = _median_flow(u, v, f)
        print(f"[pyramid vectors] {a.kind} {a.W}x{a.H} {a.levels} levels -> {os.path.abspath(a.vectors)};"
              f" median flow ({mu:.2f}, {mv:.2f}), flag fraction {fr:.2f}, stats count={st['count']}")
        raise SystemExit
    print(f"{a.W}x{a.H}, {a.levels}-level pyramid, RY={RY}")
    print(f"{'true (dx,dy)':>16} | {'single-scale LK':>18} | {'pyramid LK':>18} | valid fraction")
    for dx, dy in [(0.4, -0.3), (1.5, 1.0), (3.2, -2.1), (5.6, 3.3), (9.3, -6.2), (14.7, 8.1)]:
        I0, I1 = shifted_pair(a.W, a.H, dx, dy, seed=3)
        _, _, (u1, v1, f1) = G.lk_fixed(I0, I1)
        u, v, f = lk_pyramid_fixed(I0, I1, a.levels)
        su, sv, _ = _median_flow(u1, v1, f1)
        pu, pv, fr = _median_flow(u, v, f)
        print(f"({dx:5.1f},{dy:5.1f})    | ({su:7.2f},{sv:7.2f}) | ({pu:7.2f},{pv:7.2f}) | {fr:.2f}")
        st = frame_stats(u, v, f)
    print("statistics registers of the last case:", st)
