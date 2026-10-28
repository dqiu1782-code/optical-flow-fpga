"""
prof_speed.py -- why CPU / GPU / FPGA speeds differ so much: break the time down

Measures (pyramid LK, 5 levels, RY=30, same PyTorch implementation as bench_pyr.py):
  1. Per-stage timing: pyramid build, and per level upsample guess / warp / gradient / 5x5 window / solve, on CPU and GPU
     (synchronised around each stage, so the sum is slightly larger than unsynchronised)
  2. Op count and memory traffic: TorchDispatchMode records how many aten ops one pair calls
     and how many bytes each reads/writes (view ops count as no traffic). Single-scale LK is recorded too for comparison.
  3. GPU profiler: CUDA kernel launches per pair, and kernel run time as a share of wall time
  4. Micro-benchmarks: fixed cost of a single small op on GPU/CPU (launch overhead), and memory bandwidth of a large copy

Usage:  python python/prof_speed.py --sintel E:\\sintel
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lk_golden as G                          # noqa: E402
from bench_cpu_gpu import lk_torch             # noqa: E402
from bench_full import load_all                # noqa: E402
from bench_pyr import pyr_lk_torch             # noqa: E402

LEVELS, RY = 5, 30


# ------------------------------------------------------------------ 1. per-stage timing
def staged(I0, I1, det_min, sync, rec, levels=LEVELS, ry=RY):
    def mark(name, t0):
        sync()
        t = time.perf_counter()
        rec[name] += t - t0
        return t

    sync()
    t = time.perf_counter()
    P0, P1 = [I0], [I1]
    for _ in range(levels - 1):
        P0.append(F.avg_pool2d(P0[-1][:, None], 2)[:, 0])
        P1.append(F.avg_pool2d(P1[-1][:, None], 2)[:, 0])
    t = mark("pyramid build", t)
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
        t = mark(f"L{l} upsample guess", t)
        yy, xx = torch.meshgrid(torch.arange(H, device=a.device, dtype=a.dtype),
                                torch.arange(W, device=a.device, dtype=a.dtype), indexing="ij")
        gx = (xx + gu).clamp(0, W - 1) / (W - 1) * 2 - 1
        gy = (yy + gv).clamp(0, H - 1) / (H - 1) * 2 - 1
        bw = F.grid_sample(b[:, None], torch.stack([gx, gy], -1), mode="bilinear",
                           padding_mode="border", align_corners=True)[:, 0]
        t = mark(f"L{l} warp", t)
        Ix = torch.zeros_like(a)
        Iy = torch.zeros_like(a)
        It = torch.zeros_like(a)
        Ix[:, 1:-1, 1:-1] = a[:, 1:-1, 2:] - a[:, 1:-1, :-2]
        Iy[:, 1:-1, 1:-1] = a[:, 2:, 1:-1] - a[:, :-2, 1:-1]
        It[:, 1:-1, 1:-1] = bw[:, 1:-1, 1:-1] - a[:, 1:-1, 1:-1]
        t = mark(f"L{l} gradient", t)
        P = torch.stack([Ix * Ix, Ix * Iy, Iy * Iy, Ix * It, Iy * It], dim=1)
        S = F.avg_pool2d(P, kernel_size=5, stride=1) * 25.0
        S = F.pad(S, (2, 2, 2, 2))
        t = mark(f"L{l} 5x5 window", t)
        Sxx, Sxy, Syy, Sxt, Syt = S.unbind(1)
        det = Sxx * Syy - Sxy * Sxy
        flag = det > det_min
        safe = torch.where(flag, det, torch.ones_like(det))
        du = torch.where(flag, -2.0 * (Syy * Sxt - Sxy * Syt) / safe, torch.zeros_like(det))
        dv = torch.where(flag, -2.0 * (Sxx * Syt - Sxy * Sxt) / safe, torch.zeros_like(det))
        u, v = gu + du, gv + dv
        t = mark(f"L{l} solve", t)
    l1 = (u.abs() + v.abs()) * flag
    st = (l1.sum((1, 2)) / flag.sum((1, 2)).clamp(min=1)).cpu()
    mark("stats + copy back", t)
    return st


def run_staged(frames_t, i0, i1, batch, dm, sync, npairs):
    rec = defaultdict(float)
    for s in range(0, npairs, batch):
        staged(frames_t[i0[s:s + batch]].float(), frames_t[i1[s:s + batch]].float(), dm, sync, rec)
    return {k: v / npairs * 1e3 for k, v in rec.items()}       # ms per pair


# ------------------------------------------------------------------ 2. op count + memory traffic
VIEW_OPS = {"view", "_unsafe_view", "slice", "select", "expand", "as_strided", "unsqueeze", "squeeze",
            "permute", "t", "transpose", "unbind", "split", "alias", "detach", "_reshape_alias", "reshape",
            "lift_fresh", "unfold"}
FACTORY_OPS = {"zeros_like", "ones_like", "empty_like", "full_like", "arange", "zeros", "empty", "ones",
               "full", "scalar_tensor", "empty_strided"}


def count_ops(fn):
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    stats = {"ops": 0, "traffic_ops": 0, "bytes": 0, "by_op": defaultdict(lambda: [0, 0])}

    def nbytes(x):
        return x.numel() * x.element_size() if isinstance(x, torch.Tensor) else 0

    class M(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            name = func.overloadpacket.__name__.rstrip("_")
            stats["ops"] += 1
            if name in VIEW_OPS:
                return out
            ins = sum(nbytes(x) for x in tree_flatten((args, kwargs or {}))[0]) if name not in FACTORY_OPS else 0
            outs = sum(nbytes(x) for x in tree_flatten(out)[0])
            stats["traffic_ops"] += 1
            stats["bytes"] += ins + outs
            stats["by_op"][name][0] += 1
            stats["by_op"][name][1] += ins + outs
            return out

    with M():
        fn()
    stats["by_op"] = dict(sorted(((k, v) for k, v in stats["by_op"].items()), key=lambda kv: -kv[1][1]))
    return stats


# ------------------------------------------------------------------ 4. micro-benchmarks
def micro(dev, sync):
    r = {}
    x = torch.zeros(16, device=dev)
    for _ in range(200):
        x.add_(1)
    sync()
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        x.add_(1)
    sync()
    r["small_op_us"] = (time.perf_counter() - t0) / n * 1e6
    big = torch.empty(256 * 1024 * 1024 // 4 * 4, device=dev)       # 1 GiB float32
    dst = torch.empty_like(big)
    dst.copy_(big)
    sync()
    reps = 10 if dev.type == "cuda" else 3
    t0 = time.perf_counter()
    for _ in range(reps):
        dst.copy_(big)
    sync()
    r["copy_GBps"] = 2 * big.numel() * 4 * reps / (time.perf_counter() - t0) / 1e9   # read + write
    del big, dst
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel", required=True)
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--out", default="results_speed_profile.json")
    a = ap.parse_args()
    dm = float(G.DET_MIN)
    pairs, frames, i0, i1 = load_all(a.sintel)
    i0t, i1t = torch.from_numpy(i0), torch.from_numpy(i1)
    res = {"levels": LEVELS, "ry": RY, "pairs_profiled": a.pairs, "torch": torch.__version__}

    # ---- op count / traffic (one pair, device-independent)
    f0 = torch.from_numpy(frames[i0[:1]]).float()
    f1 = torch.from_numpy(frames[i1[:1]]).float()
    res["ops_pyramid"] = count_ops(lambda: pyr_lk_torch(f0, f1, LEVELS, RY, dm))
    res["ops_single"] = count_ops(lambda: lk_torch(f0, f1, dm))
    for k in ("ops_pyramid", "ops_single"):
        s = res[k]
        print(f"[{k}] {s['ops']} aten ops ({s['traffic_ops']} move data), "
              f"{s['bytes'] / 1e6:.1f} MB read+written per pair", flush=True)

    # ---- CPU
    torch.set_num_threads(os.cpu_count())
    fc = torch.from_numpy(frames)
    nop = lambda: None
    run_staged(fc, i0t, i1t, 1, dm, nop, 4)
    res["cpu_micro"] = micro(torch.device("cpu"), nop)
    res["cpu_stage_ms_batch1"] = run_staged(fc, i0t, i1t, 1, dm, nop, min(a.pairs, 32))
    print(f"[CPU] small op {res['cpu_micro']['small_op_us']:.2f} us, copy bandwidth {res['cpu_micro']['copy_GBps']:.1f} GB/s, "
          f"stages total {sum(res['cpu_stage_ms_batch1'].values()):.2f} ms/pair", flush=True)

    # ---- GPU
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        sync = torch.cuda.synchronize
        fg = fc.to(dev)
        g0, g1 = i0t.to(dev), i1t.to(dev)
        run_staged(fg, g0, g1, 32, dm, sync, 64)
        res["gpu_micro"] = micro(dev, sync)
        res["gpu_stage_ms_batch1"] = run_staged(fg, g0, g1, 1, dm, sync, a.pairs)
        res["gpu_stage_ms_batch32"] = run_staged(fg, g0, g1, 32, dm, sync, max(a.pairs, 256))
        print(f"[GPU] small op {res['gpu_micro']['small_op_us']:.2f} us, copy bandwidth {res['gpu_micro']['copy_GBps']:.0f} GB/s, "
              f"stages total batch1 {sum(res['gpu_stage_ms_batch1'].values()):.3f} ms/pair, "
              f"batch32 {sum(res['gpu_stage_ms_batch32'].values()):.3f} ms/pair", flush=True)
        # profiler: kernel count, kernel time vs wall time
        from torch.profiler import profile, ProfilerActivity
        for B, n in ((1, 16), (32, 64)):
            for s in range(0, 64, B):          # warm-up
                pyr_lk_torch(fg[g0[s:s + B]].float(), fg[g1[s:s + B]].float(), LEVELS, RY, dm)
            sync()
            t0 = time.perf_counter()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for s in range(0, n, B):
                    pyr_lk_torch(fg[g0[s:s + B]].float(), fg[g1[s:s + B]].float(), LEVELS, RY, dm)
                sync()
            wall = time.perf_counter() - t0
            ev = prof.key_averages()
            kern_us = sum(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) for e in ev if getattr(e, "device_type", None) is not None
                          and str(e.device_type).endswith("CUDA"))
            launches = sum(e.count for e in ev if e.key in ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel"))
            top = sorted(((e.key, e.count, getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))) for e in ev
                          if str(getattr(e, "device_type", "")).endswith("CUDA")), key=lambda x: -x[2])[:12]
            res[f"gpu_prof_batch{B}"] = {"pairs": n, "kernel_launches_per_pair": launches / n,
                                         "kernel_time_ms_per_pair": kern_us / 1e3 / n,
                                         "wall_ms_per_pair_profiled": wall * 1e3 / n,
                                         "top_kernels": top}
            print(f"[GPU profiler batch{B}] {launches / n:.0f} kernel launches per pair, kernel run time "
                  f"{kern_us / 1e3 / n:.3f} ms, wall {wall * 1e3 / n:.3f} ms (profiler adds overhead)", flush=True)

    res["ops_pyramid"]["by_op"] = {k: v for k, v in list(res["ops_pyramid"]["by_op"].items())[:25]}
    res["ops_single"]["by_op"] = {k: v for k, v in list(res["ops_single"]["by_op"].items())[:25]}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print(f"results written to {a.out}")


if __name__ == "__main__":
    main()
