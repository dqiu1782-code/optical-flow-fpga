"""
synth_report.py -- resource estimate with the open-source Yosys (synth_xilinx), and how many lanes fit on one chip

v2: default target is the K26 on the Kria KV260 (Zynq UltraScale+, -family xcup), and both designs are synthesised:
  single   = lk_top                       single-scale LK (the original three-stage pipeline, WMAX=1024)
  pyramid  = lk_level + lk_pyramid        one pyramid LK lane (warp + LK + merge + statistics + pyramid build)

Note: Yosys results differ from Vivado (usually same order of magnitude), and Yosys does no real timing analysis.
For a rough clock estimate see python/timing_est.py; re-synthesise with Vivado once a board is available.

Usage:  python python/synth_report.py                (KV260)
        python python/synth_report.py --family xc7   (XC7Z020 on the Zybo)
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICES = {
    "xcup": ("K26 (KV260)", {"LUT": 117120, "FF": 234240, "DSP": 1248, "BRAM36": 144, "URAM": 64}),
    "xc7": ("XC7Z020 (Zybo Z7-20)", {"LUT": 53200, "FF": 106400, "DSP": 220, "BRAM36": 140, "URAM": 0}),
}
# resources reserved for DMA / AXI interconnect / ARM interface
PERIPHERY = {"LUT": 15000, "FF": 20000, "DSP": 0, "BRAM36": 8, "URAM": 0}

RTL_BASE = ["rtl/lk_gradient.v", "rtl/lk_window.v", "rtl/lk_solve.v", "rtl/lk_top.v"]
TARGETS = {
    "lk_top": (RTL_BASE, "lk_top", {"WMAX": 1024}),
    "lk_level": (RTL_BASE + ["rtl/lk_warp.v", "rtl/lk_level.v"], "lk_level", {"WMAX": 1024, "NRB": 6}),
    "lk_pyramid": (["rtl/lk_pyramid.v"], "lk_pyramid", {"LEVELS": 5, "WMAX": 1024}),
}

# LUTs per LUTRAM / shift-register primitive (unknown RAM* counted as 4)
LUTRAM_LUTS = {"RAM32X1S": 1, "RAM64X1S": 1, "RAM128X1S": 2, "RAM256X1S": 4, "RAM512X1S": 8,
               "RAM32X1D": 2, "RAM64X1D": 2, "RAM128X1D": 4, "RAM256X1D": 8,
               "RAM32M": 4, "RAM64M": 4, "RAM32M16": 8, "RAM64M8": 8, "SRL16E": 1, "SRLC32E": 1}


def run_yosys(yosys, name, family, extra=""):
    files, top, params = TARGETS[name]
    ch = " ".join(f"-chparam {k} {v}" for k, v in params.items())
    script = f"""
read_verilog -sv {' '.join(files)}
hierarchy -check -top {top} {ch}
synth_xilinx -family {family} -flatten -top {top} {extra}
tee -o build_synth/stat_{name}_{family}.txt stat
"""
    path = f"build_synth/synth_{name}_{family}.ys"
    with open(path, "w") as f:
        f.write(script)
    t0 = time.perf_counter()
    r = subprocess.run([yosys, "-q", "-l", f"build_synth/synth_{name}_{family}.log", "-s", path],
                       capture_output=True, text=True)
    print(f"  {name}: {time.perf_counter() - t0:.0f} s", flush=True)
    if r.returncode != 0:
        msg = r.stdout + r.stderr
        if "-uram" in extra and ("unknown option" in msg.lower() or "syntax error" in msg.lower()):
            print(f"  {name}: this Yosys version has no -uram, re-synthesising without it (count URAM by hand)", flush=True)
            return run_yosys(yosys, name, family, extra.replace("-uram", "").strip())
        if "-nolutram" not in extra and "lutram" in msg:
            # some Yosys versions have a bug in UltraScale+ distributed RAM mapping; fall back to no LUTRAM
            print(f"  {name}: LUTRAM mapping failed, re-synthesising with -nolutram (small memories become flip-flops, resources overestimated)", flush=True)
            return run_yosys(yosys, name, family, (extra + " -nolutram").strip())
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"synthesis of {name} failed, see the output above")
    cells = {}
    for line in open(f"build_synth/stat_{name}_{family}.txt", encoding="utf-8", errors="replace"):
        m = re.match(r"\s+([A-Za-z_$][\w$]*)\s+(\d+)\s*$", line) or re.match(r"\s+(\d+)\s+([A-Za-z_$][\w$]*)\s*$", line)
        if m:
            a, b = m.groups()
            nm, cnt = (a, int(b)) if not a.isdigit() else (b, int(a))
            cells[nm] = cells.get(nm, 0) + cnt
    return cells


def summarize(cells):
    logic = sum(v for k, v in cells.items() if re.fullmatch(r"LUT[1-6]", k))
    lutram = sum(v * LUTRAM_LUTS.get(k, 4) for k, v in cells.items()
                 if (k.startswith("RAM") and not k.startswith("RAMB")) or k.startswith("SRL"))
    ff = sum(v for k, v in cells.items() if re.fullmatch(r"FD[A-Z]*", k))
    dsp = sum(v for k, v in cells.items() if k.startswith("DSP48"))
    bram = sum(v for k, v in cells.items() if re.fullmatch(r"RAMB36E\d", k)) + \
        0.5 * sum(v for k, v in cells.items() if re.fullmatch(r"RAMB18E\d", k))
    uram = sum(v for k, v in cells.items() if k.startswith("URAM"))
    return {"LUT": logic + lutram, "LUTRAM": lutram, "FF": ff, "DSP": dsp, "BRAM36": bram, "URAM": uram}


def fit(lane, dev):
    budget = {k: dev[k] * 0.8 - PERIPHERY[k] for k in dev}
    f = {k: (int(budget[k] // lane[k]) if lane[k] else 999) for k in dev}
    n = min(f.values())
    return n, min(f, key=f.get), f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="xcup", choices=list(DEVICES))
    a = ap.parse_args()
    yosys = shutil.which("yowasp-yosys") or os.path.join(os.path.dirname(sys.executable), "yowasp-yosys")
    os.chdir(ROOT)
    os.makedirs("build_synth", exist_ok=True)
    devname, dev = DEVICES[a.family]
    print(f"Yosys synth_xilinx -family {a.family}, target {devname}, a few minutes per module...", flush=True)
    # let Yosys use URAM on UltraScale+ (the warp ring line buffer is marked ram_style="ultra")
    extra = "-uram" if a.family == "xcup" else ""
    res = {name: summarize(run_yosys(yosys, name, a.family, extra)) for name in TARGETS}
    res["pyramid_lane"] = {k: res["lk_level"][k] + res["lk_pyramid"][k] for k in res["lk_level"]}

    out = {"device": devname, "family": a.family, "tool": "yowasp-yosys synth_xilinx -flatten",
           "periphery_reserved": PERIPHERY, "modules": res, "fit": {}}
    for name, title in (("lk_top", "single-scale LK (lk_top)"), ("pyramid_lane", "one pyramid LK lane (lk_level + lk_pyramid)")):
        lane = res[name]
        n, lim, f = fit(lane, dev)
        out["fit"][name] = {"max_lanes": n, "limiting": lim, "per_resource": f}
        print(f"\n{title}:")
        for k in ("LUT", "LUTRAM", "FF", "DSP", "BRAM36", "URAM"):
            tot = dev.get(k if k != "LUTRAM" else "LUT", 0)
            pct = f"{lane[k] / tot * 100:5.1f}%" if tot else "   - "
            print(f"  {k:<7} {lane[k]:>8.0f}  {pct}")
        print(f"  -> at 80% utilisation + reserved peripheral resources: up to {n} lanes (limited by {lim}); per resource {f}")
    print("\nper module:")
    for name in ("lk_top", "lk_level", "lk_pyramid"):
        print(f"  {name:<11} " + "  ".join(f"{k} {v:.0f}" for k, v in res[name].items()))
    fn = f"results_synth_{a.family}.json"
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"results written to {fn}")


if __name__ == "__main__":
    main()
