"""
pyr_eval.py -- accuracy of the fixed-point pyramid LK (Python model, bit-exact with the hardware) on all of Sintel, plus a keyframe demo

1. Accuracy: EPE (end-point error) per frame pair, single-scale vs pyramid (different level counts, vertical search limit RY).
   Also binned by ground-truth motion (<1, 1-5, 5-20, >20 px) to show which part the pyramid fixes.
2. Per-frame statistic: hardware "mean |u|+|v|" vs ground-truth mean motion, correlation.
3. Keyframe demo: per sequence, accumulate motion from the first frame and pick a keyframe when it exceeds a threshold;
   pick once with the hardware statistic and once with ground truth, then compare.

Usage:  python python/pyr_eval.py --sintel E:\\sintel --workers 24
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lk_golden as G          # noqa: E402
import lk_pyramid as P         # noqa: E402
import sintel_tools as ST      # noqa: E402

CONFIGS = {                     # name: (levels, RY, per-level residual clamp (Q8, 0 = none))
    "single": (1, 14, 0),
    "L4_RY14": (4, 14, 0),
    "L5_RY14": (5, 14, 0),      # hardware default: 5 levels, 32-line ring buffer
    "L5_RY30": (5, 30, 0),      # 64-line ring buffer
    "L6_RY14": (6, 14, 0),
    "L5_RY30_C2": (5, 30, 512),  # per-level residual clamp +-2 px
    "L5_RY30_C1": (5, 30, 256),  # per-level residual clamp +-1 px
}
MAG_BINS = [(0, 1), (1, 5), (5, 20), (20, 1e9)]
B = 2


def parse_cfg(name):
    """name -> (levels, RY, residual clamp Q8, coarse-level det threshold multiplier). Besides CONFIGS, also accepts e.g. L5_RY30_C1_D4:
    C = per-level residual clamp (px, may be fractional), D = det threshold multiplier for coarse levels (level 1 and up)"""
    import re
    if name in CONFIGS:
        L, ry, dmax = CONFIGS[name]
        return L, ry, dmax, 1
    m = re.fullmatch(r"L(\d+)_RY(\d+)(?:_C([\d.]+))?(?:_D(\d+))?", name)
    if not m:
        raise SystemExit(f"unknown config {name}")
    return int(m[1]), int(m[2]), int(float(m[3]) * 256) if m[3] else 0, int(m[4]) if m[4] else 1


def one_pair(job):
    k, (seq, f0, f1, flo), cfgs, det_min = job
    I0, I1 = ST.read_gray(f0).astype(np.int32), ST.read_gray(f1).astype(np.int32)
    gu, gv = ST.read_flo(flo)
    sl = (slice(B, -B), slice(B, -B))
    mag = np.sqrt(gu ** 2 + gv ** 2)[sl]
    gt_l1 = (np.abs(gu) + np.abs(gv))[sl]
    out = {"k": k, "seq": seq, "frame": os.path.basename(f0), "gt_mean_mag": float(mag.mean()),
           "gt_mean_l1": float(gt_l1.mean()), "zero_epe": float(mag.mean())}
    for name in cfgs:
        L, ry, dmax, dmul = parse_cfg(name)
        u, v, f = P.lk_pyramid_fixed(I0, I1, L, ry, det_min, du_max=dmax, det_mul_coarse=dmul)
        e = np.sqrt((u / 256 - gu) ** 2 + (v / 256 - gv) ** 2)[sl]
        fs = f[sl]
        st = P.frame_stats(u, v, f)
        r = {"epe_all": float(e.mean()), "epe_flag": float(e[fs].mean()) if fs.any() else float("nan"),
             "flag_frac": float(fs.mean()),
             "hw_mean_l1": st["sum_l1"] / max(st["count"], 1) / 256,
             "gt_mean_l1_on_flag": float(gt_l1[fs].mean()) if fs.any() else float("nan")}
        for lo, hi in MAG_BINS:
            m = (mag >= lo) & (mag < hi)
            r[f"epe_mag_{lo}_{int(min(hi, 999))}"] = float(e[m].mean()) if m.any() else float("nan")
            r[f"n_mag_{lo}_{int(min(hi, 999))}"] = int(m.sum())
        out[name] = r
    return out


def keyframes(scores, thr):
    """Pick the next keyframe once accumulated motion exceeds thr. scores[i] = motion from frame i to i+1. Returns keyframe indices."""
    kf, acc = [0], 0.0
    for i, s in enumerate(scores):
        acc += s
        if acc >= thr:
            kf.append(i + 1)
            acc = 0.0
    return kf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--det-min", type=int, default=G.DET_MIN)
    ap.add_argument("--kf-thr", type=float, default=30.0, help="keyframe threshold: accumulated mean |u|+|v| (px)")
    ap.add_argument("--out", default="results_pyr_eval")
    a = ap.parse_args()
    cfgs = a.configs.split(",")
    pairs = ST.list_pairs(a.sintel)
    if a.limit:
        pairs = pairs[:: max(1, len(pairs) // a.limit)][: a.limit]
    print(f"{len(pairs)} frame pairs, configs {cfgs}, {a.workers} workers", flush=True)
    t0 = time.perf_counter()
    res = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap(one_pair, [(k, p, cfgs, a.det_min) for k, p in enumerate(pairs)], chunksize=2), 1):
            res.append(r)
            if i % 100 == 0 or i == len(pairs):
                print(f"  {i}/{len(pairs)}  {time.perf_counter() - t0:.0f} s", flush=True)
    with open(a.out + ".jsonl", "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")

    summ = {"pairs": len(res), "det_min": a.det_min, "zero_epe": float(np.mean([r["zero_epe"] for r in res])), "configs": {}}
    gt = np.array([r["gt_mean_l1"] for r in res])
    for name in cfgs:
        rs = [r[name] for r in res]
        s = {k: float(np.nanmean([x[k] for x in rs])) for k in rs[0] if not k.startswith("n_mag")}
        hw = np.array([x["hw_mean_l1"] for x in rs])
        s["corr_hw_vs_gt_l1"] = float(np.corrcoef(hw, gt)[0, 1])
        rank = lambda z: np.argsort(np.argsort(z))
        s["spearman_hw_vs_gt_l1"] = float(np.corrcoef(rank(hw), rank(gt))[0, 1])
        summ["configs"][name] = s
    # keyframes per sequence: hardware statistic vs ground truth
    seqs = {}
    for r in res:
        seqs.setdefault(r["seq"], []).append(r)
    main_cfg = "L5_RY14" if "L5_RY14" in cfgs else cfgs[-1]
    kf_rows, hit, tot_gt, tot_hw = [], 0, 0, 0
    for seq, rs in seqs.items():
        rs.sort(key=lambda r: r["frame"])
        k_gt = keyframes([r["gt_mean_l1"] for r in rs], a.kf_thr)
        k_hw = keyframes([r[main_cfg]["hw_mean_l1"] for r in rs], a.kf_thr)
        k_ss = keyframes([r["single"]["hw_mean_l1"] for r in rs], a.kf_thr) if "single" in cfgs else []
        h = sum(any(abs(x - y) <= 1 for y in k_hw) for x in k_gt)
        hit, tot_gt, tot_hw = hit + h, tot_gt + len(k_gt), tot_hw + len(k_hw)
        kf_rows.append({"seq": seq, "frames": len(rs) + 1, "kf_gt": k_gt, "kf_hw": k_hw, "kf_single": k_ss})
    summ["keyframes"] = {"threshold_px": a.kf_thr, "config": main_cfg, "n_gt": tot_gt, "n_hw": tot_hw,
                         "gt_kf_matched_within_1": hit, "match_rate": hit / max(tot_gt, 1),
                         "n_single": sum(len(r["kf_single"]) for r in kf_rows), "per_seq": kf_rows}
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print(f"\nzero-flow baseline EPE {summ['zero_epe']:.3f}")
    for name, s in summ["configs"].items():
        print(f"{name:<9} EPE all {s['epe_all']:.3f}  flag=1 {s['epe_flag']:.3f}  flag fraction {s['flag_frac']:.2f} | "
              f"by motion <1:{s['epe_mag_0_1']:.2f} 1-5:{s['epe_mag_1_5']:.2f} 5-20:{s['epe_mag_5_20']:.2f} >20:{s['epe_mag_20_999']:.2f} | "
              f"statistic vs GT corr {s['corr_hw_vs_gt_l1']:.3f} (Spearman {s['spearman_hw_vs_gt_l1']:.3f})")
    k = summ["keyframes"]
    print(f"keyframes (threshold {a.kf_thr} px, {main_cfg}): GT picks {k['n_gt']}, hardware statistic picks {k['n_hw']}, "
          f"{k['match_rate'] * 100:.0f}% of GT keyframes matched by hardware within +-1 frame; single-scale picks {k['n_single']}")
    print(f"results: {a.out}.json / .jsonl")


if __name__ == "__main__":
    main()
