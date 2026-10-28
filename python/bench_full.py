"""
bench_full.py -- the whole Sintel training set (clean, all 1041 pairs) on CPU and GPU: total time + accuracy

Difference from bench_cpu_gpu.py: that one samples frames to measure per-frame speed; this one really runs the whole dataset end to end,
reports how long the entire dataset takes, and computes accuracy on all 1041 pairs.

Timing (same as the FPGA): inputs are already-decoded 8-bit grey images; the uint8 -> float conversion is timed;
PNG reading is not. For the GPU an end-to-end time including copying the whole dataset from RAM to VRAM is also reported.

Usage:  python python/bench_full.py --sintel E:\\sintel
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lk_golden as G                          # noqa: E402
import sintel_tools as ST                      # noqa: E402
from bench_cpu_gpu import lk_torch, GpuPowerSampler   # noqa: E402


def load_all(root):
    """Read each image once (adjacent pairs share images). Returns frames (uint8 NxHxW), pair indices, GT paths."""
    pairs = ST.list_pairs(root)
    paths = sorted({p[1] for p in pairs} | {p[2] for p in pairs})
    index = {p: i for i, p in enumerate(paths)}
    frames = np.stack([ST.read_gray(p) for p in paths])
    i0 = np.array([index[p[1]] for p in pairs])
    i1 = np.array([index[p[2]] for p in pairs])
    return pairs, frames, i0, i1


def run_all(frames_t, i0, i1, batch, det_min, sync):
    """Run LK over all pairs and return wall time (s)."""
    n = len(i0)
    sync()
    t0 = time.perf_counter()
    for s in range(0, n, batch):
        a = frames_t[i0[s:s + batch]].float()
        b = frames_t[i1[s:s + batch]].float()
        lk_torch(a, b, det_min)
    sync()
    return time.perf_counter() - t0


def epe_torch(u, v, gu, gv, flag, b=2):
    sl = (slice(b, -b), slice(b, -b))
    u, v, gu, gv, flag = (x[sl] for x in (u, v, gu, gv, flag))
    e = torch.sqrt((u - gu) ** 2 + (v - gv) ** 2)
    small = torch.sqrt(gu ** 2 + gv ** 2) < 1.0
    fs = flag & small
    return {"epe_all": e.mean().item(),
            "epe_flag": e[flag].mean().item() if flag.any() else float("nan"),
            "flag_frac": flag.float().mean().item(),
            "epe_small_flag": e[fs].mean().item() if fs.any() else float("nan"),
            "zero_epe_all": torch.sqrt(gu ** 2 + gv ** 2).mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel", required=True)
    ap.add_argument("--det-min", type=int, default=G.DET_MIN)
    ap.add_argument("--out", default="results_full_cpu_gpu.json")
    a = ap.parse_args()
    dm = float(a.det_min)

    t = time.perf_counter()
    pairs, frames, i0, i1 = load_all(a.sintel)
    N, (H, W) = len(pairs), frames.shape[1:]
    print(f"all {N} pairs ({len(frames)} distinct images), {W}x{H}, loading {time.perf_counter() - t:.1f}s")
    res = {"pairs": N, "resolution": [W, H], "det_min": a.det_min, "torch": torch.__version__}

    # ------------------------------------------------------------ CPU
    torch.set_num_threads(os.cpu_count())
    fc = torch.from_numpy(frames)
    i0t, i1t = torch.from_numpy(i0), torch.from_numpy(i1)
    run_all(fc, i0t[:20], i1t[:20], 1, dm, lambda: None)           # warm-up
    for B in (1, 16):
        ts = [run_all(fc, i0t, i1t, B, dm, lambda: None) for _ in range(2)]
        res[f"cpu_total_s_batch{B}"] = min(ts)
        print(f"[CPU {torch.get_num_threads()} threads, batch={B}] all {N} pairs {min(ts):.2f} s "
              f"= {min(ts) / N * 1e3:.2f} ms per pair ({N / min(ts):.0f} fps)")

    # ------------------------------------------------------------ GPU
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
        run_all(fg, g0[:64], g1[:64], 32, dm, sync)                  # warm-up
        for B in (1, 32):
            ts = [run_all(fg, g0, g1, B, dm, sync) for _ in range(3)]
            res[f"gpu_total_s_batch{B}"] = min(ts)
            print(f"[GPU {res['gpu_name']}, batch={B}] all {N} pairs {min(ts):.3f} s "
                  f"= {min(ts) / N * 1e3:.3f} ms per pair ({N / min(ts):.0f} fps)")
        res["gpu_end_to_end_s"] = res["gpu_upload_s"] + res["gpu_total_s_batch32"]
        print(f"      copy whole dataset RAM -> VRAM {res['gpu_upload_s']:.3f} s; end to end (copy + batch32) {res['gpu_end_to_end_s']:.3f} s")
        with GpuPowerSampler() as pw:
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 6:
                run_all(fg, g0, g1, 32, dm, sync)
        res["gpu_power_w_batch32"] = statistics.median(pw.samples) if pw.samples else None
        with GpuPowerSampler() as pw:
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 6:
                run_all(fg, g0, g1, 1, dm, sync)
        res["gpu_power_w_batch1"] = statistics.median(pw.samples) if pw.samples else None
        print(f"      power (nvidia-smi median): batch1 {res['gpu_power_w_batch1']} W, batch32 {res['gpu_power_w_batch32']} W")

        # ---------------------------------------------------------- accuracy on all 1041 pairs
        stats = []
        for k, (_, _, _, flo) in enumerate(pairs):
            u, v, f = lk_torch(fg[g0[k:k + 1]].float(), fg[g1[k:k + 1]].float(), dm)
            gu, gv = ST.read_flo(flo)
            stats.append(epe_torch(u[0], v[0], torch.from_numpy(gu).to(dev), torch.from_numpy(gv).to(dev), f[0]))
        res["epe_float_all_pairs"] = {key: float(np.nanmean([s[key] for s in stats])) for key in stats[0]}
        print(f"accuracy on all {N} pairs (float32): {res['epe_float_all_pairs']}")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"results written to {a.out}")


if __name__ == "__main__":
    main()
