"""sanity_sintel.py -- two baselines that help interpret the accuracy numbers on Sintel
1) Zero-flow baseline: EPE if nothing is computed and (0,0) is output
2) Known-shift test: shift a real Sintel image by a known sub-pixel amount and check that LK recovers it
   (rules out a code bug: if this is accurate, large Sintel errors are a limit of the algorithm itself)
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sintel_tools as ST
import lk_golden as G
from bench_cpu_gpu import lk_torch

root = sys.argv[1]
pairs = ST.list_pairs(root)
sel = pairs[::max(1, len(pairs) // 60)][:60]
z_all, z_small = [], []
for _, p0, p1, flo in sel:
    gu, gv = ST.read_flo(flo)
    s = ST.epe_stats(np.zeros_like(gu), np.zeros_like(gv), gu, gv, np.ones(gu.shape, bool))
    z_all.append(s["epe_all"]); z_small.append(s["epe_small_flag"])
mag = [float(np.sqrt(ST.read_flo(p[3])[0] ** 2 + ST.read_flo(p[3])[1] ** 2).mean()) for p in sel[:20]]
print(f"zero-flow baseline: EPE(all) {np.mean(z_all):.3f}, EPE(small-motion pixels) {np.nanmean(z_small):.3f}; GT mean motion {np.mean(mag):.2f} px")

I = torch.from_numpy(ST.read_gray(sel[5][1]).astype(np.float32))
H, W = I.shape
for dx, dy in ((0.5, 0.0), (0.0, -0.3), (0.6, -0.4), (1.5, 1.0), (3.0, 0.0)):
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    grid = torch.stack([(xx - dx) / (W - 1) * 2 - 1, (yy - dy) / (H - 1) * 2 - 1], -1)[None]
    I1 = F.grid_sample(I[None, None], grid, mode="bilinear", align_corners=True)[0, 0].round()
    u, v, f = lk_torch(I[None], I1[None], float(G.DET_MIN))
    m = f[0].clone(); m[:8], m[-8:], m[:, :8], m[:, -8:] = False, False, False, False
    print(f"known shift ({dx:+.1f}, {dy:+.1f}) px -> LK median ({u[0][m].median():+.3f}, {v[0][m].median():+.3f}), "
          f"flag fraction {m.float().mean():.2f}")
