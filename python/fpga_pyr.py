"""
fpga_pyr.py -- full RTL simulation of pyramid LK on real Sintel frames (1024x436, native resolution)

For each pair: Python produces the reference -> iverilog simulates tb_lk_pyr.v (pyramid build + 5 levels + statistics)
-> checks bit-exactness and reads the clock cycles for each level and for the pyramid build (input stream never stalls, +BUBBLE=0).
Several pairs run as parallel vvp processes (only to make simulation faster; unrelated to FPGA speed).

Usage:  python python/fpga_pyr.py --sintel E:\\sintel --pairs 4
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lk_golden as G          # noqa: E402
import lk_pyramid as P         # noqa: E402
import sintel_tools as ST      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["tb/tb_lk_pyr.v", "rtl/lk_pyramid.v", "rtl/lk_level.v", "rtl/lk_warp.v", "rtl/lk_top.v",
         "rtl/lk_gradient.v", "rtl/lk_window.v", "rtl/lk_solve.v"]


def run_one(job):
    k, pair, levels, workdir, nrb, du_max = job
    wdir = os.path.join(workdir, f"p{k:04d}")
    os.makedirs(wdir, exist_ok=True)
    I0, I1 = ST.read_gray(pair[1]).astype("int32"), ST.read_gray(pair[2]).astype("int32")
    vec = os.path.join(wdir, "vectors")
    _, _, _, st = P.write_pyr_vectors(vec, I0, I1, levels, nrb, G.DET_MIN, du_max)
    vvp = os.path.join(wdir, "pyr.vvp")
    subprocess.run(["iverilog", "-g2012", "-I", vec, "-o", vvp, *[os.path.join(ROOT, f) for f in FILES]], check=True)
    t0 = time.perf_counter()
    sim = subprocess.run(["vvp", "-n", vvp, "+BUBBLE=0"], cwd=wdir, capture_output=True, text=True)
    out = sim.stdout + sim.stderr
    lv = {int(m[0]): {"W": int(m[1]), "H": int(m[2]), "outputs": int(m[3]), "errors": int(m[4]), "cycles": int(m[5])}
          for m in re.findall(r"LEVEL (\d+) (\d+)x(\d+): flow outputs (\d+) errors (\d+) .*? cycles (\d+)", out)}
    m = re.search(r"Cycles: pyramid build (\d+), all levels (\d+)", out)
    shutil.rmtree(wdir, ignore_errors=True)
    return {"k": k, "seq": pair[0], "frame": os.path.basename(pair[1]), "pass": "*** PASS ***" in out,
            "levels": lv, "pyr_cycles": int(m.group(1)) if m else None, "level_cycles": int(m.group(2)) if m else None,
            "stats": st, "sim_s": time.perf_counter() - t0, "tail": "" if "*** PASS ***" in out else out[-2500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel", required=True)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--levels", type=int, default=P.LEVELS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nrb", type=int, default=P.NRB)
    ap.add_argument("--du-max", type=int, default=P.DU_MAX)
    ap.add_argument("--workdir", default=os.path.join(ROOT, "runs_pyr"))
    ap.add_argument("--out", default="results_fpga_pyr.json")
    a = ap.parse_args()
    os.chdir(ROOT)
    pairs = ST.list_pairs(a.sintel)
    sel = pairs[:: max(1, len(pairs) // a.pairs)][: a.pairs]
    print(f"{len(sel)} real Sintel frame pairs, {a.levels}-level pyramid, {a.workers} simulation processes (a few minutes each)", flush=True)
    res = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(run_one, [(k, p, a.levels, a.workdir, a.nrb, a.du_max) for k, p in enumerate(sel)]):
            res.append(r)
            print(f"  {r['seq']}/{r['frame']}: {'PASS' if r['pass'] else 'FAIL'}  pyramid build {r['pyr_cycles']} cycles, "
                  f"levels total {r['level_cycles']} cycles  " +
                  " ".join(f"L{l}:{v['cycles']}" for l, v in sorted(r['levels'].items(), reverse=True)) +
                  f"  (sim {r['sim_s']:.0f} s)", flush=True)
            if not r["pass"]:
                print(r["tail"])
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"results written to {a.out}")


if __name__ == "__main__":
    main()
