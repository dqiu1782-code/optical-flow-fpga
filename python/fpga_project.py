"""
fpga_project.py -- how long pyramid LK on the KV260 takes for all of Sintel (1041 pairs), estimated from RTL cycle counts + synthesis resources + DDR bandwidth

Inputs:
  results_fpga_pyr.json   from fpga_pyr.py (real 1024x436 frames, per-level cycles measured in RTL simulation, input never stalls)
  results_synth_xcup.json from synth_report.py (resources of one pyramid lane, how many fit on the K26)
Model:
  - one lane = pyramid generator (lk_pyramid) + level engine (lk_level). They are separate hardware
    and can overlap: while the engine processes the levels of pair t, the generator builds the pyramid of frame t+2.
    cycles per pair = max(levels total, pyramid build) ("overlapped"); the non-overlapped figure (sequential, as in simulation) is also given.
  - DDR traffic per pair (N = 1024x436):
      pyramid build  read 1 frame N + write levels 1-4, about N/3
      levels         read I0, I1 all levels 2 x 1.33N; read coarser flow 4B x N/3; write level 1-4 flow 4B x N/3
      full res       statistics only: 0; dense flow map: another 4B x N written
    KV260 DDR4 is 19.2 GB/s in theory; 12 GB/s usable is assumed.
  - speed = min(lanes x clock / cycles per pair, bandwidth / bytes per pair)
Usage:  python python/fpga_project.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = 1041
BW = 12e9


def main():
    os.chdir(ROOT)
    sim = json.load(open("results_fpga_pyr.json", encoding="utf-8"))
    syn = json.load(open("results_synth_xcup.json", encoding="utf-8"))
    ok = [r for r in sim if r["pass"]]
    lv = max(r["level_cycles"] for r in ok)
    py = max(r["pyr_cycles"] for r in ok)
    W, H = 1024, 436
    N = W * H
    levels = sorted(ok[0]["levels"].items(), key=lambda kv: -int(kv[0]))
    # Yosys does not map a byte-write memory into one URAM (it splits it into 8); Vivado does.
    # Each of the 4 warp banks is 1024x64 bit = one URAM288, so lanes are recomputed at 4 URAMs per lane.
    lane = dict(syn["modules"]["pyramid_lane"])
    lane["URAM"] = 4
    dev = {"LUT": 117120, "FF": 234240, "DSP": 1248, "BRAM36": 144, "URAM": 64}
    peri = syn["periphery_reserved"]
    fit = {k: int((dev[k] * 0.8 - peri[k]) // lane[k]) if lane[k] else 999 for k in dev}
    lanes = min(fit.values())
    lanes_single = syn["fit"]["lk_top"]["max_lanes"]
    print(f"one pyramid lane: " + ", ".join(f"{k} {lane[k]:.0f}" for k in dev) + f" -> per resource {fit}, using {lanes}")
    frac = sum(1 / 4 ** l for l in range(1, len(levels)))          # pixels of levels 1-4 / N
    bytes_stats = N * (1 + frac) + 2 * N * (1 + frac) + 4 * N * frac * 2
    bytes_dense = bytes_stats + 4 * N
    out = {"pairs_simulated": len(sim), "pairs_pass": len(ok), "level_cycles": lv, "pyramid_cycles": py,
           "per_level": {k: v["cycles"] for k, v in levels}, "lanes": lanes, "lane_resources": lane, "fit": fit, "lanes_single_scale": lanes_single,
           "bytes_per_pair_stats": bytes_stats, "bytes_per_pair_dense": bytes_dense, "rows": []}
    print(f"RTL simulation ({len(ok)}/{len(sim)} pairs PASS): levels total {lv:,} cycles (" +
          ", ".join(f"L{k} {v['cycles']:,}" for k, v in levels) + f"), pyramid build {py:,} cycles")
    print(f"DDR traffic per pair: statistics only {bytes_stats / 1e6:.2f} MB, dense output {bytes_dense / 1e6:.2f} MB; "
          f"{lanes} pyramid lanes fit on the K26")
    for mhz in (150, 200, 250):
        for overlap in (True, False):
            cyc = max(lv, py) if overlap else lv + py
            comp = lanes * mhz * 1e6 / cyc
            for mode, b in (("stats", bytes_stats), ("dense", bytes_dense)):
                rate = min(comp, BW / b)
                row = {"mhz": mhz, "overlap": overlap, "mode": mode, "pairs_per_s": rate,
                       "dataset_s": PAIRS / rate, "bound": "compute" if comp < BW / b else "ddr"}
                out["rows"].append(row)
                print(f"  {mhz} MHz {'overlapped' if overlap else 'sequential'} {'stats only' if mode == 'stats' else 'dense output'}: "
                      f"{rate:,.0f} pairs/s -> 1041 pairs in {PAIRS / rate:.3f} s (bound: {'compute' if row['bound'] == 'compute' else 'DDR bandwidth'})")
    with open("results_fpga_project.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("results written to results_fpga_project.json")


if __name__ == "__main__":
    main()
