"""
analyze_stream.py <stream results dir> [output dir] -- figures and tables for the "FPGA/GPU/CPU working at the same time" experiments

  fig_flow_split.png       flow work split: completion time with FPGA / GPU / static split / dynamic split
  fig_gantt_<run>.png      Gantt charts (one row per device, vertical line = end of capture)
  fig_stream_vs_serial.png concurrent vs serial: total time / wait after capture ends
  stream_table.md          summary table
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LANE_C = {"FPGA": "#eda100", "GPU flow": "#e87ba4", "GPU SIFT": "#86b6ef", "GPU match": "#2a78d6", "CPU mapping": "#0d366b"}
ROWS = ["FPGA", "GPU flow", "GPU SIFT", "GPU match", "CPU mapping"]
ROWLAB = {"FPGA": "FPGA flow (10 lanes)", "GPU flow": "GPU flow", "GPU SIFT": "GPU SIFT extraction", "GPU match": "GPU matching",
          "CPU mapping": "CPU mapping"}


def base(ax, grid_axis="x"):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    (ax.xaxis if grid_axis == "x" else ax.yaxis).grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def gantt(root, name, out, title):
    p = os.path.join(root, name, "timeline.json")
    if not os.path.isfile(p):
        return
    t = json.load(open(p, encoding="utf-8"))
    ev, marks = t["events"], t["marks"]
    rows = [r for r in ROWS if any((e[0].startswith("FPGA") if r == "FPGA" else e[0] == r) for e in ev)]
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rows) + 1.5), facecolor=SURF)
    for y, r in enumerate(rows):
        segs = [(e[1], e[2] - e[1]) for e in ev if (e[0].startswith("FPGA") if r == "FPGA" else e[0] == r)]
        if r == "FPGA" and marks.get("capture_end", 0) > 0.5:
            # online: a frame pair every 33 ms and the FPGA is busy only ~1.8 ms -- too thin to draw, so a light band + text
            s0, s1 = min(x[0] for x in segs), max(x[0] + x[1] for x in segs)
            ax.broken_barh([(s0, s1 - s0)], (y - 0.32, 0.64), facecolors=LANE_C[r], alpha=0.25, edgecolor="none")
            ax.text((s0 + s1) / 2, y, "statistics within 1.8 ms of each frame (computed during capture)", ha="center", va="center", fontsize=8.5,
                    color="#8a5a00")
        elif r == "FPGA":
            ax.broken_barh(segs, (y - 0.32, 0.64), facecolors=LANE_C[r], edgecolor="none")
        else:
            segs = [(s, max(w, 0.0004 * marks.get("end", 1))) for s, w in segs]     # widen very short segments so they are visible
            ax.broken_barh(segs, (y - 0.32, 0.64), facecolors=LANE_C[r], edgecolor="none")
        busy = sum(e[2] - e[1] for e in ev if (e[0].startswith("FPGA") if r == "FPGA" else e[0] == r))
        if r == "FPGA":
            busy /= 10.0
        ax.text(marks["end"] * 1.005, y, f" busy {100 * busy / max(marks['end'], 1e-9):.0f}%", va="center", fontsize=7.5,
                color=MUTED)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([ROWLAB[r] for r in rows], fontsize=9)
    ax.invert_yaxis()
    base(ax)
    if marks.get("capture_end", 0) > 0.5:
        ax.axvline(marks["capture_end"], color="#c0392b", lw=1.2, ls="--")
        ax.text(marks["capture_end"], -0.6, "end of capture ", color="#c0392b", fontsize=8.5, va="bottom", ha="right")
    ax.axvline(marks["end"], color=INK2, lw=1)
    ax.text(marks["end"], -0.6, f" done {marks['end']:.1f} s", color=INK2, fontsize=8.5, va="bottom")
    ax.set_xlabel("time (s)", color=INK2, fontsize=9)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=18)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    root = sys.argv[1]
    outd = sys.argv[2] if len(sys.argv) > 2 else root
    S = json.load(open(os.path.join(root, "summary.json"), encoding="utf-8"))
    # ---- A: flow split
    keys = [("A_flow_fpga", "FPGA only"), ("A_flow_gpu", "GPU only"), ("A_flow_split_static", "static split"),
            ("A_flow_split_dynamic", "dynamic split (idle device takes next)")]
    keys = [(k, l) for k, l in keys if k in S]
    if keys:
        fig, ax = plt.subplots(figsize=(8, 2.6), facecolor=SURF)
        for i, (k, lab) in enumerate(keys):
            r = S[k]
            v = r["flow_end"] * 1000
            ax.barh(i, v, color=["#eda100", "#e87ba4", "#9b7fd1", "#2a78d6"][i], height=0.6)
            extra = f"  FPGA {r['n_pairs_fpga']} / GPU {r['n_pairs_gpu']} pairs" if r["n_pairs_gpu"] and r["n_pairs_fpga"] else ""
            ax.text(v, i, f" {v:.0f} ms{extra}", va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([l for _, l in keys], fontsize=9)
        ax.invert_yaxis()
        base(ax)
        ax.set_xlabel("time to finish flow + statistics for 2584 frame pairs (ms)", color=INK2, fontsize=9)
        ax.set_title("Splitting the flow work between FPGA and GPU (TUM fr3, 640×480, offline)", fontsize=10.5, color=INK, loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(outd, "fig_flow_split.png"), dpi=160)
        plt.close(fig)
    # ---- Gantt charts
    for name in S:
        if name.startswith(("B_", "C_")):
            r = S[name]
            mode = "concurrent (pipelined)" if not r["serial"] else "serial"
            rate = "online 30 fps" if r["rate"] == "online" else "offline"
            gantt(root, name, os.path.join(outd, f"fig_gantt_{name}.png"),
                  f"{r['method']} K={r['K']} · {rate} · {mode} · flow={r['flow']}")
    # ---- concurrent vs serial
    pairs = []
    for name, r in S.items():
        if name.startswith("B_") and name.endswith("_stream"):
            ser = name[:-len("_stream")] + "_serial"
            if ser in S:
                pairs.append((name[2:-7], r, S[ser]))
    if pairs:
        pairs.sort(key=lambda x: (x[1]["rate"], x[1]["method"], x[1]["K"]))
        fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(pairs) + 1.6), facecolor=SURF, sharey=True)
        for ax, key, t in zip(axes, ("total_s", "after_capture_s"), ("total time (s)", "wait after capture ends (s)")):
            for i, (lab, a, b) in enumerate(pairs):
                ax.barh(i - 0.18, b[key], height=0.34, color=BASE, label="serial" if i == 0 else None)
                ax.barh(i + 0.18, a[key], height=0.34, color="#2a78d6", label="concurrent" if i == 0 else None)
                ax.text(a[key], i + 0.18, f" {a[key]:.0f}", va="center", fontsize=7.5, color=INK2)
                ax.text(b[key], i - 0.18, f" {b[key]:.0f}", va="center", fontsize=7.5, color=MUTED)
            base(ax)
            ax.set_title(t, fontsize=10.5, color=INK, loc="left")
        axes[0].set_yticks(range(len(pairs)))
        axes[0].set_yticklabels([p[0].replace("_", " ") for p in pairs], fontsize=8.5)
        axes[0].invert_yaxis()
        axes[0].legend(fontsize=8.5, frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(outd, "fig_stream_vs_serial.png"), dpi=160)
        plt.close(fig)
    # ---- table
    L = ["| run | method | K | flow | mode | flow done s | capture end s | total s | wait after capture s | mapping batches | registered | ATE mm |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in S.items():
        ate = (r.get("ate") or {}).get("rmse")
        L.append(f"| {name} | {r['method']} | {r['K']} | {r['flow']} | {'serial' if r['serial'] else 'concurrent'} | "
                 f"{r['flow_end']:.3f} | {r['marks'].get('capture_end', 0):.1f} | {r['total_s']:.1f} | "
                 f"{r['after_capture_s']:.1f} | {r.get('n_map_chunks', '–')} | {r.get('registered', '–')} | "
                 f"{'–' if ate is None else f'{ate * 1000:.1f}'} |")
    open(os.path.join(outd, "stream_table.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
