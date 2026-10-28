"""
bench_cpu_gpu.py -- speed + accuracy of the same Lucas-Kanade algorithm on CPU (9950X) and GPU (RTX 5090)

The algorithm is identical to the FPGA one (central-difference gradients, 5x5 window, closed-form 2x2 solve, same det threshold),
only float32 on CPU/GPU vs fixed-point integers on the FPGA. One PyTorch implementation runs on both cpu and cuda,
so the comparison is the same computation on different hardware.

Timing (same for all three platforms):
  - Only the flow computation is timed. PNG reading and grey conversion are excluded (the FPGA input is also a ready grey pixel stream).
  - For the GPU, a time including the CPU->GPU copy is reported separately, since that step cannot be avoided in practice.
  - Warm up first, then repeat and take the median; the GPU calls torch.cuda.synchronize() before and after each timing.

Usage:
  python python/bench_cpu_gpu.py --sintel D:\\sintel --pairs 200
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lk_golden as G          # noqa: E402
import sintel_tools as ST      # noqa: E402

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("PyTorch missing. GPU build: python -m pip install torch --index-url https://download.pytorch.org/whl/cu128")


def lk_torch(I0, I1, det_min):
    """I0, I1: (B, H, W) float32, values 0..255. Returns u, v (px), flag."""
    B, H, W = I0.shape
    Ix = torch.zeros_like(I0)
    Iy = torch.zeros_like(I0)
    It = torch.zeros_like(I0)
    Ix[:, 1:-1, 1:-1] = I0[:, 1:-1, 2:] - I0[:, 1:-1, :-2]
    Iy[:, 1:-1, 1:-1] = I0[:, 2:, 1:-1] - I0[:, :-2, 1:-1]
    It[:, 1:-1, 1:-1] = I1[:, 1:-1, 1:-1] - I0[:, 1:-1, 1:-1]
    P = torch.stack([Ix * Ix, Ix * Iy, Iy * Iy, Ix * It, Iy * It], dim=1)       # (B,5,H,W)
    S = F.avg_pool2d(P, kernel_size=5, stride=1) * 25.0                         # (B,5,H-4,W-4)
    S = F.pad(S, (2, 2, 2, 2))                                                   # zero-pad a 2-pixel border
    Sxx, Sxy, Syy, Sxt, Syt = S.unbind(1)
    det = Sxx * Syy - Sxy * Sxy
    flag = det > det_min
    safe = torch.where(flag, det, torch.ones_like(det))
    u = torch.where(flag, -2.0 * (Syy * Sxt - Sxy * Syt) / safe, torch.zeros_like(det))
    v = torch.where(flag, -2.0 * (Sxx * Syt - Sxy * Sxt) / safe, torch.zeros_like(det))
    return u, v, flag


class GpuPowerSampler:
    """Background thread that reads GPU power with nvidia-smi every 100 ms."""
    def __init__(self):
        self.samples, self._stop = [], threading.Event()

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=2).stdout.strip().splitlines()
                if out:
                    self.samples.append(float(out[0]))
            except Exception:
                pass
            time.sleep(0.1)

    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.t.join()


def time_it(fn, sync, reps, min_seconds=2.0):
    """Warm up 3 times, then repeat at least reps times and at least min_seconds; returns per-run times (s)."""
    for _ in range(3):
        fn()
    sync()
    ts, t_start = [], time.perf_counter()
    while len(ts) < reps or time.perf_counter() - t_start < min_seconds:
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel", required=True, help="Sintel root after unzipping (contains training/)")
    ap.add_argument("--pairs", type=int, default=200, help="number of pairs for timing (evenly sampled)")
    ap.add_argument("--epe-pairs", type=int, default=40, help="number of pairs for accuracy (the fixed-point model is slow)")
    ap.add_argument("--det-min", type=int, default=G.DET_MIN)
    ap.add_argument("--batch", type=int, default=32, help="batch size for GPU throughput")
    ap.add_argument("--out", default="results_cpu_gpu.json")
    a = ap.parse_args()

    pairs = ST.list_pairs(a.sintel)
    if not pairs:
        raise SystemExit(f"training/clean and training/flow not found under {a.sintel}")
    step = max(1, len(pairs) // a.pairs)
    sel = pairs[::step][:a.pairs]
    print(f"Sintel clean: {len(pairs)} pairs, {len(sel)} used for timing")

    t0 = time.perf_counter()
    F0 = np.stack([ST.read_gray(p[1]) for p in sel])
    F1 = np.stack([ST.read_gray(p[2]) for p in sel])
    N, H, W = F0.shape
    print(f"loaded {N} pairs, {W}x{H}, {time.perf_counter() - t0:.1f}s")

    res = {"resolution": [W, H], "pairs_timed": N, "det_min": a.det_min,
           "cpu_name": platform.processor(), "torch": torch.__version__}

    # ------------------------------------------------------------ CPU
    torch.set_num_threads(os.cpu_count())
    c0 = torch.from_numpy(F0).float()
    c1 = torch.from_numpy(F1).float()
    k = [0]

    def cpu_one():
        i = k[0] % N
        k[0] += 1
        lk_torch(c0[i:i + 1], c1[i:i + 1], float(a.det_min))
    ts = time_it(cpu_one, lambda: None, reps=N)
    res["cpu_ms_per_frame"] = statistics.median(ts) * 1e3
    res["cpu_threads"] = torch.get_num_threads()
    print(f"[CPU  {res['cpu_threads']} threads] {res['cpu_ms_per_frame']:.2f} ms per pair  "
          f"({1e3 / res['cpu_ms_per_frame']:.0f} fps)")

    # ------------------------------------------------------------ GPU
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        res["gpu_name"] = torch.cuda.get_device_name(0)
        g0 = c0.to(dev)
        g1 = c1.to(dev)
        sync = torch.cuda.synchronize

        def gpu_one():
            i = k[0] % N
            k[0] += 1
            lk_torch(g0[i:i + 1], g1[i:i + 1], float(a.det_min))

        def gpu_one_with_copy():
            i = k[0] % N
            k[0] += 1
            lk_torch(c0[i:i + 1].to(dev, non_blocking=False), c1[i:i + 1].to(dev, non_blocking=False),
                     float(a.det_min))

        B = min(a.batch, N)

        def gpu_batch():
            i = (k[0] * B) % max(1, N - B + 1)
            k[0] += 1
            lk_torch(g0[i:i + B], g1[i:i + B], float(a.det_min))

        with GpuPowerSampler() as pw:
            ts = time_it(gpu_one, sync, reps=N, min_seconds=3.0)
        res["gpu_ms_per_frame"] = statistics.median(ts) * 1e3
        res["gpu_power_w"] = statistics.median(pw.samples) if pw.samples else None
        ts = time_it(gpu_one_with_copy, sync, reps=N)
        res["gpu_ms_per_frame_with_copy"] = statistics.median(ts) * 1e3
        with GpuPowerSampler() as pw:
            ts = time_it(gpu_batch, sync, reps=20, min_seconds=3.0)
        res["gpu_batch"] = B
        res["gpu_ms_per_frame_batched"] = statistics.median(ts) * 1e3 / B
        res["gpu_power_w_batched"] = statistics.median(pw.samples) if pw.samples else None
        print(f"[GPU {res['gpu_name']}] single frame {res['gpu_ms_per_frame']:.3f} ms | with copy "
              f"{res['gpu_ms_per_frame_with_copy']:.3f} ms | batch={B} per frame {res['gpu_ms_per_frame_batched']:.4f} ms")
        print(f"      power (nvidia-smi median): single-frame {res['gpu_power_w']} W, batched {res['gpu_power_w_batched']} W")
    else:
        print("[GPU] torch.cuda.is_available() = False, skipping GPU (probably a CPU-only PyTorch build)")

    # ------------------------------------------------------------ accuracy (EPE)
    estep = max(1, len(pairs) // a.epe_pairs)
    esel = pairs[::estep][:a.epe_pairs]
    acc_f, acc_q = [], []
    # consistency check: torch version vs the Python reference (float64) on the first pair should be nearly identical
    I0, I1 = ST.read_gray(esel[0][1]), ST.read_gray(esel[0][2])
    (_, _, _), S_ref, _ = G.lk_fixed(I0, I1, a.det_min)
    u_ref, v_ref, f_ref = G.solve_flow(S_ref, a.det_min)
    u_t, v_t, f_t = lk_torch(torch.from_numpy(I0).float()[None], torch.from_numpy(I1).float()[None], float(a.det_min))
    agree = (f_t[0].numpy() == f_ref)
    both = f_ref & f_t[0].numpy()
    res["sanity_flag_agreement"] = float(agree.mean())
    res["sanity_median_abs_diff_px"] = float(np.median(np.abs(u_t[0].numpy()[both] - u_ref[both]))) if both.any() else None
    print(f"consistency check: flag agreement {res['sanity_flag_agreement']:.5f}, median |u| difference {res['sanity_median_abs_diff_px']}")
    for seq, p0, p1, flo in esel:
        I0, I1 = ST.read_gray(p0), ST.read_gray(p1)
        gu, gv = ST.read_flo(flo)
        u, v, fl = lk_torch(torch.from_numpy(I0).float()[None], torch.from_numpy(I1).float()[None], float(a.det_min))
        acc_f.append(ST.epe_stats(u[0].numpy(), v[0].numpy(), gu, gv, fl[0].numpy()))
        _, _, (uq, vq, fq) = G.lk_fixed(I0, I1, a.det_min)          # FPGA fixed-point model (bit-exact with RTL)
        acc_q.append(ST.epe_stats(uq / 256.0, vq / 256.0, gu, gv, fq))

    def avg(lst):
        return {k: float(np.nanmean([d[k] for d in lst])) for k in lst[0]}
    res["epe_float"] = avg(acc_f)
    res["epe_fixed_fpga"] = avg(acc_q)
    print(f"accuracy ({len(esel)} pairs): float32 {res['epe_float']}")
    print(f"              FPGA fixed-point {res['epe_fixed_fpga']}")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"results written to {a.out}")


if __name__ == "__main__":
    main()
