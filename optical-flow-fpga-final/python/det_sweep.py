"""det_sweep.py -- effect of the det threshold (DET_MIN) on accuracy / coverage (runs on GPU, a few seconds)
Usage:  python python/det_sweep.py --sintel E:\\sintel
"""
import argparse, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sintel_tools as ST
from bench_cpu_gpu import lk_torch

ap = argparse.ArgumentParser()
ap.add_argument("--sintel", required=True)
ap.add_argument("--pairs", type=int, default=60)
a = ap.parse_args()
pairs = ST.list_pairs(a.sintel)
sel = pairs[::max(1, len(pairs) // a.pairs)][:a.pairs]
dev = "cuda" if torch.cuda.is_available() else "cpu"
data = []
for _, p0, p1, flo in sel:
    gu, gv = ST.read_flo(flo)
    data.append((torch.from_numpy(ST.read_gray(p0)).float()[None].to(dev),
                 torch.from_numpy(ST.read_gray(p1)).float()[None].to(dev), gu, gv))
print(f"{len(data)} frame pairs, device {dev}")
print(f"{'DET_MIN':>12} {'flag frac':>9} {'EPE(all)':>10} {'EPE(flag)':>10} {'EPE(flag,small)':>16}")
for k in (0, 8, 12, 14, 16, 18, 20, 22, 24):
    dm = 0 if k == 0 else 1 << k
    st = []
    for I0, I1, gu, gv in data:
        u, v, f = lk_torch(I0, I1, float(dm))
        st.append(ST.epe_stats(u[0].cpu().numpy(), v[0].cpu().numpy(), gu, gv, f[0].cpu().numpy()))
    m = {key: float(np.nanmean([s[key] for s in st])) for key in st[0]}
    print(f"{dm:>12} {m['flag_frac']:>9.3f} {m['epe_all']:>10.3f} {m['epe_flag']:>10.3f} {m['epe_small_flag']:>16.3f}")
