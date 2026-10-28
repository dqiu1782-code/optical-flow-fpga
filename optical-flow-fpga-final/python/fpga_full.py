"""
fpga_full.py -- the whole Sintel training set (all 1041 pairs) through RTL simulation of the multi-lane FPGA design

One simulation = LANES pipelines on one FPGA processing LANES frame pairs at once. The 1041 pairs are split into ceil(1041/LANES) batches.
For each batch:
  1. the Python fixed-point model produces the reference
  2. vvp runs the RTL simulation, checks every pixel of every lane for bit-exactness, and reads the exact cycle count
  3. the fixed-point result (proven bit-exact with the RTL) is compared with Sintel ground truth for accuracy
Simulation is slow (minutes per batch), so several vvp processes run in parallel -- only to finish the simulation sooner,
unrelated to FPGA speed. FPGA speed = cycles / clock frequency.

Resumable: results are appended per batch to .jsonl, and a rerun skips finished batches.

Usage:  python python/fpga_full.py --sintel E:\\sintel --lanes 4 --workers 16
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lk_golden as G          # noqa: E402
import sintel_tools as ST      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RTL = ["tb/tb_lk_multi.v", "rtl/lk_multi.v", "rtl/lk_top.v", "rtl/lk_gradient.v", "rtl/lk_window.v", "rtl/lk_solve.v"]


def run_group(job):
    gi, group, vvp, workdir, det_min = job
    wdir = os.path.join(workdir, f"g{gi:04d}")
    os.makedirs(wdir, exist_ok=True)
    frame_pairs = [(ST.read_gray(p[1]), ST.read_gray(p[2])) for p in group]
    res = G.write_multi_vectors(os.path.join(wdir, "vectors"), frame_pairs, det_min)
    t0 = time.perf_counter()
    sim = subprocess.run(["vvp", "-n", vvp], cwd=wdir, capture_output=True, text=True)
    sim_s = time.perf_counter() - t0
    out = sim.stdout + sim.stderr
    lanes = [dict(zip(("lane", "outputs", "errors", "flag"), map(int, m)))
             for m in re.findall(r"LANE (\d+) outputs (\d+) errors (\d+) flag (\d+)", out)]
    m = re.search(r"Cycles first pixel in -> last output of all lanes: (\d+)", out)
    stats = []
    for k, p in enumerate(group):
        gu, gv = ST.read_flo(p[3])
        uq, vq, fq = res[k]
        e = ST.epe_stats(uq / 256.0, vq / 256.0, gu, gv, fq)
        e["zero_epe_all"] = ST.epe_stats(np.zeros_like(gu), np.zeros_like(gv), gu, gv, fq)["epe_all"]
        stats.append({"seq": p[0], "frame": os.path.basename(p[1]), **e})
    shutil.rmtree(wdir, ignore_errors=True)
    return {"group": gi, "pass": "*** PASS ***" in out, "lanes": lanes,
            "cycles": int(m.group(1)) if m else None, "sim_s": sim_s, "pairs": stats,
            "tail": "" if "*** PASS ***" in out else out[-2000:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel", required=True)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16, help="parallel simulation processes (only affects simulation speed)")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N batches (for a trial)")
    ap.add_argument("--det-min", type=int, default=G.DET_MIN)
    ap.add_argument("--workdir", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--out", default="results_fpga_full")
    a = ap.parse_args()
    os.chdir(ROOT)

    pairs = ST.list_pairs(a.sintel)
    H, W = ST.read_gray(pairs[0][1]).shape
    L = a.lanes
    groups = []
    for s in range(0, len(pairs), L):
        g = pairs[s:s + L]
        groups.append(g + [g[-1]] * (L - len(g)))      # pad the last batch with repeated pairs (removed in the statistics)
    n_real = [min(L, len(pairs) - s) for s in range(0, len(pairs), L)]
    if a.limit:
        groups = groups[:a.limit]

    # compile once (all batches have the same image size)
    build = os.path.join(ROOT, f"build_multi{L}")
    os.makedirs(build, exist_ok=True)
    G.write_params(build, W, H, a.det_min, lanes=L)
    vvp = os.path.join(build, "multi.vvp")
    subprocess.run(["iverilog", "-g2012", "-I", build, "-o", vvp, *RTL], check=True)

    jl = a.out + ".jsonl"
    done = {}
    if os.path.exists(jl):
        for line in open(jl, encoding="utf-8"):
            r = json.loads(line)
            done[r["group"]] = r
    todo = [(gi, g, vvp, a.workdir, a.det_min) for gi, g in enumerate(groups) if gi not in done]
    print(f"{len(pairs)} pairs -> {len(groups)} batches x {L} lanes; {len(done)} done, {len(todo)} to run, "
          f"{a.workers} simulation processes", flush=True)

    t0 = time.perf_counter()
    with Pool(a.workers) as pool, open(jl, "a", encoding="utf-8") as f:
        for k, r in enumerate(pool.imap_unordered(run_group, todo), 1):
            r["pairs"] = r["pairs"][:n_real[r["group"]]]
            done[r["group"]] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            el = time.perf_counter() - t0
            print(f"[{len(done)}/{len(groups)}] batch {r['group']}: {'PASS' if r['pass'] else 'FAIL'}, "
                  f"{r['cycles']} cycles, sim {r['sim_s']:.0f}s | elapsed {el / 60:.1f} min, "
                  f"about {el / k * (len(todo) - k) / 60:.1f} min left", flush=True)
            if not r["pass"]:
                print(r["tail"])

    # ---------------- summary ----------------
    rs = [done[g] for g in sorted(done)]
    allp = [p for r in rs for p in r["pairs"]]
    cyc = [r["cycles"] for r in rs if r["cycles"]]
    STEADY = (W * H) + 3 * W + 4                     # cycles per frame in continuous operation: pixels + two FLUSH stages (1024x436 measured 449,540)
    summ = {
        "pairs": len(allp), "groups": len(rs), "lanes": L, "resolution": [W, H], "det_min": a.det_min,
        "groups_pass": sum(r["pass"] for r in rs),
        "pixel_errors": sum(l["errors"] for r in rs for l in r["lanes"]),
        "pixels_checked": sum(l["outputs"] for r in rs for l in r["lanes"]),
        "cycles_per_group_min": min(cyc), "cycles_per_group_max": max(cyc),
        "steady_cycles_per_frame": STEADY,
    }
    nb = math.ceil(len(allp) / L)
    for mhz in (100, 150):
        summ[f"dataset_s_{L}lanes_{mhz}MHz"] = nb * STEADY / (mhz * 1e6)
        summ[f"dataset_s_1lane_{mhz}MHz"] = len(allp) * STEADY / (mhz * 1e6)
    for key in ("epe_all", "epe_flag", "flag_frac", "epe_small_flag", "zero_epe_all"):
        summ["mean_" + key] = float(np.nanmean([p[key] for p in allp]))
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print(json.dumps(summ, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
