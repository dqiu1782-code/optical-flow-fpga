"""
timing_est.py -- rough timing estimate without Vivado: longest combinational path between registers

Method:
  1. Yosys synthesises the design to a word-level netlist (adders, multipliers, muxes, registers, RAM; not yet mapped to LUTs)
     and writes JSON.
  2. This script treats the netlist as a graph: registers and synchronous-read RAMs are start/end points, combinational cells are edges,
     each operation gets an empirical UltraScale+ delay (K26, -1 speed grade), and the longest path is found.
  3. Estimated Fmax ~= 1 / longest path delay.

Empirical delays (including routing, conservative): one LUT level 0.45 ns (single-bit and/or/mux 0.25 ns, since Vivado packs several into one LUT6); add/sub/compare 0.55 + 0.11 ns per 8 bits (CARRY8);
DSP multiply (no internal registers) 1.8 ns; Block RAM clock-to-out 1.3 ns; plus 0.4 ns per path
(clock-to-q + setup + clock skew).
**This is an estimate, not timing analysis.** Error may be +-30%; the Vivado timing report is what counts.
It is most useful for comparison: with the same method, what are the longest paths of v1 and v2 and where are they.

Usage:  python python/timing_est.py                  (default: lk_top and lk_level in rtl/)
        python python/timing_est.py --v1             (also analyse the old version in rtl_v1/ for comparison)
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LUT = 0.45
LUT_SMALL = 0.25
PATH_OVERHEAD = 0.4
BRAM_CO = 1.3
BRAM_SETUP = 0.4


def adder(w):
    return 0.55 + 0.11 * math.ceil(max(w, 1) / 8)


def lut_tree(w, fanin=6):
    return LUT * max(1, math.ceil(math.log(max(w, 2)) / math.log(fanin)))


def cell_delay(c):
    t = c["type"]
    p = {k: int(v, 2) if isinstance(v, str) and set(v) <= {"0", "1"} else v for k, v in c.get("parameters", {}).items()}
    aw, bw, yw = p.get("A_WIDTH", 1), p.get("B_WIDTH", 1), p.get("Y_WIDTH", 1)
    if t in ("$add", "$sub", "$neg", "$alu", "$lt", "$le", "$gt", "$ge", "$lcu"):
        return adder(max(aw, bw, yw if t in ("$add", "$sub", "$neg", "$alu") else 1))
    if t == "$mul":
        small, big = sorted((aw, bw))
        if small <= 4:
            return adder(max(aw, bw)) + LUT
        if small <= 18 and big <= 27:
            return 1.8
        return 1.8 + adder(aw + bw)              # split over several DSPs, then add
    if t in ("$logic_not", "$not", "$pos", "$buf"):
        return 0.0                               # inversion folds into the next LUT
    if yw == 1 and t in ("$and", "$or", "$xor", "$logic_and", "$logic_or", "$mux"):
        return LUT_SMALL                         # small single-bit logic: about two per LUT6
    if t in ("$eq", "$ne", "$eqx", "$nex", "$reduce_and", "$reduce_or", "$reduce_xor", "$reduce_bool",
             "$logic_and", "$logic_or"):
        return lut_tree(aw + (bw if t in ("$eq", "$ne", "$eqx", "$nex", "$logic_and", "$logic_or") else 0))
    if t in ("$mux", "$bmux"):
        return LUT
    if t == "$pmux":
        return LUT * (1 + math.ceil(math.log(max(p.get("S_WIDTH", 2), 2), 4)))
    if t in ("$and", "$or", "$xor", "$xnor"):
        return LUT
    if t in ("$not", "$pos", "$buf"):
        return 0.0
    if t in ("$shl", "$shr", "$sshl", "$sshr", "$shift", "$shiftx"):
        return LUT * max(1, math.ceil(math.log2(max(bw, 2)) / 2))
    return LUT


def is_seq(t):
    return "dff" in t or t in ("$ff", "$dlatch", "$adlatch", "$sr") or t.startswith("$_DFF") or t.startswith("$_SDFF")


SEQ_IN = {"D", "EN", "SRST", "ARST", "CLR", "SET", "AD", "ALOAD"}


def analyse(json_path, top, topn=6):
    mod = json.load(open(json_path))["modules"][top]
    names = {}
    for n, d in mod["netnames"].items():
        for b in d["bits"]:
            if isinstance(b, int) and (b not in names or (names[b].startswith("$") and not n.startswith("$"))):
                names[b] = n
    arrival = {}            # bit -> (arrival time, predecessor bit, cells passed)
    starts, ends = set(), []    # ends: (bit, extra delay, end-point description)
    comb = []               # (cell, inputs, outputs, delay)
    for cn, c in mod["cells"].items():
        t = c["type"]
        dirs, conn = c["port_directions"], c["connections"]
        ins = [b for pn, bits in conn.items() if dirs.get(pn) == "input" for b in bits if isinstance(b, int)]
        outs = [b for pn, bits in conn.items() if dirs.get(pn) == "output" for b in bits if isinstance(b, int)]
        if is_seq(t):
            for b in outs:
                starts.add(b)
                arrival[b] = (0.0, None, cn)
            for pn, bits in conn.items():
                if dirs.get(pn) == "input" and pn in SEQ_IN:
                    for b in bits:
                        if isinstance(b, int):
                            ends.append((b, 0.0, f"register {cn}"))
        elif t in ("$mem_v2", "$mem"):
            p = c["parameters"]
            nrd = int(p.get("RD_PORTS", "1"), 2) if isinstance(p.get("RD_PORTS"), str) else int(p.get("RD_PORTS", 1))
            ce = p.get("RD_CLK_ENABLE", "1")
            ce = int(ce, 2) if isinstance(ce, str) else int(ce)
            sync = ce == (1 << nrd) - 1
            depth = int(p["SIZE"], 2) if isinstance(p.get("SIZE"), str) else int(p.get("SIZE", 1))
            if sync:
                for b in conn.get("RD_DATA", []):
                    if isinstance(b, int):
                        starts.add(b)
                        arrival[b] = (BRAM_CO, None, cn)
                for pn in ("RD_ADDR", "RD_EN", "WR_ADDR", "WR_DATA", "WR_EN"):
                    for b in conn.get(pn, []):
                        if isinstance(b, int):
                            ends.append((b, BRAM_SETUP, f"RAM {cn}"))
            else:
                d = 0.9 + LUT * max(0, math.ceil(math.log(max(depth, 64) / 64, 4)))
                ins_r = [b for b in conn.get("RD_ADDR", []) if isinstance(b, int)]
                comb.append((cn, ins_r, [b for b in conn.get("RD_DATA", []) if isinstance(b, int)], d, "LUTRAM read"))
                for pn in ("WR_ADDR", "WR_DATA", "WR_EN"):
                    for b in conn.get(pn, []):
                        if isinstance(b, int):
                            ends.append((b, 0.3, f"RAM {cn}"))
        else:
            comb.append((cn, ins, outs, cell_delay(c), t))
    # longest path in topological order
    by_in = defaultdict(list)
    indeg = {}
    for i, (cn, ins, outs, d, t) in enumerate(comb):
        ins_c = [b for b in set(ins) if b not in starts]
        indeg[i] = 0
        for b in ins_c:
            by_in[b].append(i)
    producer = {}
    for i, (cn, ins, outs, d, t) in enumerate(comb):
        for b in outs:
            producer[b] = i
    for i, (cn, ins, outs, d, t) in enumerate(comb):
        indeg[i] = len({b for b in ins if b in producer and b not in starts})
    ready = [i for i, k in indeg.items() if k == 0]
    done = 0
    while ready:
        i = ready.pop()
        done += 1
        cn, ins, outs, d, t = comb[i]
        best, pb = 0.0, None
        for b in ins:
            if b in arrival and arrival[b][0] >= best:
                best, pb = arrival[b][0], b
        for b in outs:
            arrival[b] = (best + d, pb, f"{t}:{cn}")
        for b in set(outs):
            for j in by_in.get(b, []):
                indeg[j] -= 1
                if indeg[j] == 0:
                    ready.append(j)
    paths = []
    for b, extra, what in ends:
        if b in arrival:
            paths.append((arrival[b][0] + extra + PATH_OVERHEAD, b, what))
    paths.sort(key=lambda x: -x[0])
    out, seen = [], set()
    for tot, b, what in paths:
        if what in seen:
            continue
        seen.add(what)
        chain, cur, guard = [], b, 0
        while cur is not None and guard < 200:
            a, prev, via = arrival[cur]
            chain.append(f"{names.get(cur, cur)}<-{via.split(':')[0]}")
            cur, guard = prev, guard + 1
        out.append({"delay_ns": round(tot, 2), "fmax_mhz": round(1000 / tot), "end": what,
                    "path": " | ".join(reversed(chain[:12]))})
        if len(out) >= topn:
            break
    return out, done, len(comb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", action="store_true", help="also analyse rtl_v1/ (old version) for comparison")
    a = ap.parse_args()
    yosys = shutil.which("yowasp-yosys") or os.path.join(os.path.dirname(sys.executable), "yowasp-yosys")
    os.chdir(ROOT)
    os.makedirs("build_synth", exist_ok=True)
    jobs = [("v2 lk_top", "rtl", ["lk_gradient.v", "lk_window.v", "lk_solve.v", "lk_top.v"], "lk_top", "-chparam WMAX 1024"),
            ("v2 lk_level (one pyramid-level engine)", "rtl", ["lk_gradient.v", "lk_window.v", "lk_solve.v", "lk_top.v", "lk_warp.v", "lk_level.v"],
             "lk_level", "-chparam WMAX 1024 -chparam NRB 6"),
            ("v2 lk_pyramid", "rtl", ["lk_pyramid.v"], "lk_pyramid", "-chparam LEVELS 5 -chparam WMAX 1024")]
    if a.v1:
        jobs.insert(0, ("v1 lk_top", "rtl_v1", ["lk_gradient.v", "lk_window.v", "lk_solve.v", "lk_top.v"], "lk_top",
                        "-chparam W 1024 -chparam H 436"))
    res = {}
    for title, d, files, top, ch in jobs:
        tag = title.split()[0] + "_" + top
        js = f"build_synth/word_{tag}.json"
        script = f"""
read_verilog -sv {' '.join(os.path.join(d, f) for f in files)}
hierarchy -check -top {top} {ch}
proc
flatten
opt -full
wreduce
opt_clean
memory -nomap
opt_clean
write_json {js}
"""
        with open(f"build_synth/word_{tag}.ys", "w") as f:
            f.write(script)
        r = subprocess.run([yosys, "-q", "-s", f"build_synth/word_{tag}.ys"], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            raise SystemExit(f"{title} failed")
        paths, ndone, ncomb = analyse(js, top)
        res[title] = paths
        print(f"\n===== {title} =====  ({ncomb} combinational cells, {ndone} sorted)")
        for p in paths:
            print(f"  {p['delay_ns']:5.2f} ns  ~= {p['fmax_mhz']:4d} MHz   end point {p['end']}")
            print(f"        {p['path']}")
    with open("results_timing_est.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\nResults written to results_timing_est.json (rough estimate; the Vivado timing report is what counts)")


if __name__ == "__main__":
    main()
