"""
bench_pyr.py -- the same pyramid LK algorithm on CPU (9950X) and GPU (RTX 5090) over all of Sintel (1041 pairs)

Same algorithm as the FPGA version (2x2 average downsampling, previous-level flow x2 as the guess, vertical guess clamped to +-RY, bilinear warp,
same three-stage LK), only in float32 on CPU/GPU. Timing follows bench_full.py: inputs are already-decoded
8-bit grey images, PNG reading is not timed; for the GPU an end-to-end time including the upload to VRAM and power are also reported.
Also outputs the per-frame statistic (mean |u|+|v|) -- the GPU only has to copy these few numbers back.

Usage:  python python/bench_pyr.py --sintel E:\\sintel
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lk_golden as G                          # noqa: E402
import sintel_tools as ST                      # noqa: E402
from bench_cpu_gpu import lk_torch, GpuPowerSampler   # noqa: E402
from bench_full import load_all, epe_torch     # noqa: E402


def pyr_lk_torch(I0, I1, levels=5, ry=14, det_min=65536.0):
    """I0, I1: (B,H,W) float32. Returns full-resolution u, v (px), flag, and per-frame statistic mean(|u|+|v|) over flag."""
    P0, P1 = [I0], [I1]
    for _ in range(levels - 1):
        P0.append(F.avg_pool2d(P0[-1][:, None], 2)[:, 0])     # floor size, same as the hardware
        P1.append(F.avg_pool2d(P1[-1][:, None], 2)[:, 0])
    u = v = None
    for l in range(levels - 1, -1, -1):
        a, b = P0[l], P1[l]
        Bn, H, W = a.shape
        if u is None:
            gu = torch.zeros_like(a)
            gv = torch.zeros_like(a)
        else:
            Hc, Wc = u.shape[1:]
            ys = torch.clamp(torch.arange(H, device=a.device) // 2, max=Hc - 1)
            xs = torch.clamp(torch.arange(W, device=a.device) // 2, max=Wc - 1)
            gu = (2 * u)[:, ys][:, :, xs]
            gv = torch.clamp((2 * v)[:, ys][:, :, xs], -ry, ry)
        yy, xx = torch.meshgrid(torch.arange(H, device=a.device, dtype=a.dtype),
                                torch.arange(W, device=a.device, dtype=a.dtype), indexing="ij")
        gx = (xx + gu).clamp(0, W - 1) / (W - 1) * 2 - 1
        gy = (yy + gv).clamp(0, H - 1) / (H - 1) * 2 - 1
        bw = F.grid_sample(b[:, None], torch.stack([gx, gy], -1), mode="bilinear",
                           padding_mode="border", align_corners=True)[:, 0]
        du, dv, flag = lk_torch(a, bw, det_min)
        u, v = gu + du, gv + dv
    l1 = (u.abs() + v.abs()) * flag
    stat = l1.sum((1, 2)) / flag.sum((1, 2)).clamp(min=1)
    return u, v, flag, stat


def run_all(frames_t, i0, i1, batch, levels, ry, dm, sync):
    n = len(i0)
    sync()
    t0 = time.perf_counter()
    for s in range(0, n, batch):
        a = frames_t[i0[s:s + batch]].float()
        b = frames_t[i1[s:s + batch]].float()
        _, _, _, st = pyr_lk_torch(a, b, levels, ry, dm)
        st.cpu()                            # copy back only the per-frame statistic
    sync()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel", required=True)
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--ry", type=int, default=14)
    ap.add_argument("--det-min", type=int, default=G.DET_MIN)
    ap.add_argument("--out", default="results_pyr_cpu_gpu.json")
    a = ap.parse_args()
    dm = float(a.det_min)
    pairs, frames, i0, i1 = load_all(a.sintel)
    N = len(pairs)
    res = {"pairs": N, "levels": a.levels, "ry": a.ry, "torch": torch.__version__}
    i0t, i1t = torch.from_numpy(i0), torch.from_numpy(i1)

    torch.set_num_threads(os.cpu_count())
    fc = torch.from_numpy(frames)
    run_all(fc, i0t[:8], i1t[:8], 1, a.levels, a.ry, dm, lambda: None)
    for Bt in (1, 16):
        ts = [run_all(fc, i0t, i1t, Bt, a.levels, a.ry, dm, lambda: None) for _ in range(2)]
        res[f"cpu_total_s_batch{Bt}"] = min(ts)
        print(f"[CPU {torch.get_num_threads()} threads, batch={Bt}] pyramid LK, all {N} pairs {min(ts):.2f} s "
              f"= {min(ts) / N * 1e3:.2f} ms per pair", flush=True)

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        res["gpu_name"] = torch.cuda.get_device_name(0)
        sync = torch.cuda.synchronize
        g0, g1 = i0t.to(dev), i1t.to(dev)
        sync()
        t0 = time.perf_counter()
        fg = fc.to(dev)
        sync()
        res["gpu_upload_s"] = time.perf_counter() - t0
        run_all(fg, g0[:64], g1[:64], 32, a.levels, a.ry, dm, sync)
        for Bt in (1, 32):
            ts = [run_all(fg, g0, g1, Bt, a.levels, a.ry, dm, sync) for _ in range(3)]
            res[f"gpu_total_s_batch{Bt}"] = min(ts)
            print(f"[GPU {res['gpu_name']}, batch={Bt}] pyramid LK, all {N} pairs {min(ts):.3f} s "
                  f"= {min(ts) / N * 1e3:.3f} ms per pair", flush=True)
        for Bt in (32, 1):
            with GpuPowerSampler() as pw:
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < 6:
                    run_all(fg, g0, g1, Bt, a.levels, a.ry, dm, sync)
            res[f"gpu_power_w_batch{Bt}"] = statistics.median(pw.samples) if pw.samples else None
        print(f"      upload to VRAM {res['gpu_upload_s']:.3f} s; power batch1 {res['gpu_power_w_batch1']} W, "
              f"batch32 {res['gpu_power_w_batch32']} W", flush=True)
        stats = []
        for k, (_, _, _, flo) in enumerate(pairs):
            u, v, f, _ = pyr_lk_torch(fg[g0[k:k + 1]].float(), fg[g1[k:k + 1]].float(), a.levels, a.ry, dm)
            gu, gv = ST.read_flo(flo)
            stats.append(epe_torch(u[0], v[0], torch.from_numpy(gu).to(dev), torch.from_numpy(gv).to(dev), f[0]))
        res["epe_float_pyr"] = {key: float(np.nanmean([s[key] for s in stats])) for key in stats[0]}
        print(f"float32 pyramid LK accuracy (all {N} pairs): {res['epe_float_pyr']}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"results written to {a.out}")


if __name__ == "__main__":
    main()
