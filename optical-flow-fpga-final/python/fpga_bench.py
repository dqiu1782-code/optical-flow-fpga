"""
fpga_bench.py -- full RTL simulation on real Sintel frames: exact FPGA clock-cycle counts, from which hardware time is estimated

Why "estimated": there is no FPGA board yet. The simulator (iverilog) is tens of thousands of times slower than real hardware, so
simulation wall time means nothing; but the cycle count is exact -- real hardware does the same thing in every cycle.
    FPGA time = cycles / clock frequency
The clock is only known after Vivado synthesis; here two typical values for this kind of design on a Zynq-7020 are used: 100 MHz and 150 MHz.

Also checks that the RTL output on real Sintel images is bit-exact with the Python fixed-point model, pixel by pixel.

Usage:  python python/fpga_bench.py --sintel D:\\sintel --pair 0
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lk_golden as G          # noqa: E402
import sintel_tools as ST      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RTL = ["rtl/lk_top.v", "rtl/lk_gradient.v", "rtl/lk_window.v", "rtl/lk_solve.v"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintel", required=True)
    ap.add_argument("--pair", type=int, default=0, help="which frame pair to use (list_pairs order)")
    ap.add_argument("--det-min", type=int, default=G.DET_MIN)
    ap.add_argument("--out", default="results_fpga.json")
    a = ap.parse_args()
    os.chdir(ROOT)

    pairs = ST.list_pairs(a.sintel)
    seq, p0, p1, flo = pairs[a.pair]
    I0, I1 = ST.read_gray(p0), ST.read_gray(p1)
    H, W = I0.shape
    print(f"Sintel {seq} {os.path.basename(p0)} -> {os.path.basename(p1)}, {W}x{H}")

    print("generating test vectors (Python fixed-point model)...", flush=True)
    _, _, (uq, vq, fq) = G.write_all_vectors("vectors", I0, I1, a.det_min)

    subprocess.run(["iverilog", "-g2012", "-I", "vectors", "-o", "top.vvp", "tb/tb_lk_top.v", *RTL], check=True)
    print("running RTL simulation (two full frames, tens of seconds to a few minutes)...", flush=True)
    t0 = time.perf_counter()
    sim = subprocess.run(["vvp", "-n", "top.vvp", "+BUBBLE=0", "+NFRAMES=2"], capture_output=True, text=True)
    wall = time.perf_counter() - t0
    out = sim.stdout + sim.stderr
    print("\n".join(l for l in out.splitlines() if l.strip()))

    lat = int(re.search(r"latency: (\d+) cycles", out).group(1))
    ss = int(re.search(r"Steady-state: (\d+) cycles", out).group(1))
    passed = "*** PASS ***" in out
    res = {"sequence": seq, "frame": os.path.basename(p0), "resolution": [W, H], "det_min": a.det_min,
           "rtl_bit_exact_pass": passed, "cycles_latency_one_frame": lat, "cycles_per_frame_steady": ss,
           "sim_wallclock_s": wall}
    for mhz in (100, 150):
        res[f"fpga_ms_per_frame_{mhz}MHz"] = ss / (mhz * 1e3)
        res[f"fpga_latency_ms_{mhz}MHz"] = lat / (mhz * 1e3)
        print(f"@{mhz} MHz: {ss / (mhz * 1e3):.3f} ms per frame ({mhz * 1e6 / ss:.0f} fps), "
              f"single-frame latency {lat / (mhz * 1e3):.3f} ms")
    gu, gv = ST.read_flo(flo)
    res["epe_fixed"] = ST.epe_stats(uq / 256.0, vq / 256.0, gu, gv, fq)
    print(f"accuracy on this pair: {res['epe_fixed']}")
    print(f"(the simulation itself took {wall:.1f} s wall time -- unrelated to FPGA speed)")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"results written to {a.out}")


if __name__ == "__main__":
    main()
