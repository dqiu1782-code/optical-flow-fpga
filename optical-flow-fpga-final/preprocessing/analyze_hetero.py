"""
analyze_hetero.py <results dir>  -- summarise the report.json of each experiment (from Windows) into tables + figures

Results directory layout (json copied from E:\\hetero\\work):
  fr3/report.json            K=200, incremental mapping (+3DGS)
  fr3_K100|400|800/          K scaling, incremental
  fr3_glo_K200|400|800/      global mapping (global_mapper)
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MC = {"B1": "#898781", "B2": "#c3a35a", "O1": "#86b6ef", "O2": "#2a78d6", "O2@1.0": "#5598e7", "O2@1.5": "#1c5cab"}
LABEL = {"B1": "B1 uniform + exhaustive", "B2": "B2 uniform + sequential", "O1": "O1 flow keyframes + exhaustive", "O2": "O2 flow keyframes + pruning",
         "O2@1.0": "O2 window 1.0", "O2@1.5": "O2 window 1.5"}
STAGE = [("fpga_s", "FPGA flow statistics", "#eda100"), ("select_s", "ARM selection/pruning", "#e87ba4"),
         ("extract_s", "GPU feature extraction", "#86b6ef"), ("match_s", "GPU matching", "#2a78d6"), ("map_s", "CPU mapping", "#0d366b")]


def load(root):
    out = {}
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, "report.json")
        if not os.path.isfile(p):
            continue
        raw = open(p, "rb").read()
        try:
            rows = json.loads(raw.decode("utf-8"))["rows"]
        except UnicodeDecodeError:          # on Windows, hetero_prep wrote report.json in the default GBK encoding
            rows = json.loads(raw.decode("gbk"))["rows"]
        if not rows or not isinstance(rows[0], dict) or not d.startswith("fr3"):
            continue
        mapper = "global" if "glo" in d else "incremental"
        K = int(d.split("K")[-1]) if "_K" in d else 200
        for r in rows:
            out[(mapper, K, r["method"])] = r
    return out


def base(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def fig_scaling(R, out, mapper):
    Ks = sorted({k for (m, k, _) in R if m == mapper})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), facecolor=SURF)
    for meth in ("B1", "B2", "O1", "O2", "O2@1.5"):
        xs = [k for k in Ks if (mapper, k, meth) in R]
        if not xs:
            continue
        rr = [R[(mapper, k, meth)] for k in xs]
        kw = dict(color=MC[meth], marker="o", lw=2 if meth == "O2" else 1.4, label=LABEL[meth])
        axes[0].plot(xs, [r["prep_total_s"] for r in rr], **kw)
        axes[1].plot(xs, [r["pairs"] for r in rr], **kw)
        ate = [(r["ate_m"] if r["ate_m"] is not None else float("nan")) * 1000 for r in rr]
        axes[2].plot(xs, [a if a < 40 else float("nan") for a in ate], **kw)
        for x, a in zip(xs, ate):
            if a >= 40:      # mapping failed (trajectory collapsed) -- marked at the top of the plot
                axes[2].annotate(f"{a:.0f} mm", (x, 39), color=MC[meth], fontsize=8, ha="center", va="top")
                axes[2].plot([x], [39.5], marker="^", color=MC[meth])
    for ax, t in zip(axes, ("total preprocessing time (s)", "matched image pairs", "ATE (mm)")):
        base(ax)
        ax.set_xscale("log")
        ax.set_xticks(Ks)
        ax.set_xticklabels([str(k) for k in Ks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("number of keyframes K", color=INK2, fontsize=9)
        ax.set_title(t, fontsize=10.5, color=INK, loc="left")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    for ax in axes[:2]:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[2].set_ylim(0, 40)
    axes[0].legend(fontsize=8.5, frameon=False)
    name = "incremental mapping (mapper)" if mapper == "incremental" else "global mapping (global_mapper)"
    fig.suptitle(f"TUM fr3_long_office_household: K scaling, {name}", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_breakdown(R, out, K):
    keys = [(mp, meth) for mp in ("incremental", "global") for meth in ("B1", "B2", "O1", "O2", "O2@1.5")
            if (mp, K, meth) in R]
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(keys) + 1.4), facecolor=SURF)
    for i, (mp, meth) in enumerate(keys):
        r = R[(mp, K, meth)]
        left = 0
        for k, lab, c in STAGE:
            v = r[k] or 0
            ax.barh(i, v, left=left, color=c, height=0.62, label=lab if i == 0 else None)
            left += v
        ax.text(left + 1, i, f"{left:.1f} s", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{LABEL[m]} ({'incr.' if mp == 'incremental' else 'global'})" for mp, m in keys], fontsize=9)
    ax.invert_yaxis()
    base(ax)
    ax.set_xlabel("seconds", color=INK2, fontsize=9)
    ax.legend(ncol=5, fontsize=8.5, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0))
    ax.set_title(f"K={K} time per stage (FPGA from RTL cycles, 10 lanes @ 250 MHz)", fontsize=10.5, color=INK, loc="left", pad=24)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def table(R):
    lines = ["| mapper | K | method | pairs | loops | FPGA | select | extract | match | map | total (s) | registered | ATE (mm) | PSNR | SSIM |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (mp, K, meth), r in sorted(R.items(), key=lambda x: (x[0][0] != "incremental", x[0][1], x[0][2])):
        f = lambda v, d=1: "–" if v is None else f"{v:.{d}f}"
        lines.append(f"| {'incremental' if mp == 'incremental' else 'global'} | {K} | {meth} | {r['pairs']} | {r['loops']} | "
                     f"{f(r['fpga_s'], 2)} | {f(r['select_s'], 3)} | {f(r['extract_s'])} | {f(r['match_s'])} | "
                     f"{f(r['map_s'])} | {f(r['prep_total_s'])} | {r['registered']}/{r['K']} | "
                     f"{f(None if r['ate_m'] is None else r['ate_m'] * 1000)} | {f(r.get('psnr'), 2)} | {f(r.get('ssim'), 3)} |")
    return "\n".join(lines)


def main():
    root = sys.argv[1]
    outd = sys.argv[2] if len(sys.argv) > 2 else root
    R = load(root)
    for mp in ("incremental", "global"):
        if any(k[0] == mp for k in R):
            fig_scaling(R, os.path.join(outd, f"fig_scaling_{mp}.png"), mp)
    for K in sorted({k[1] for k in R}):
        fig_breakdown(R, os.path.join(outd, f"fig_breakdown_K{K}.png"), K)
    t = table(R)
    open(os.path.join(outd, "summary_table.md"), "w", encoding="utf-8").write(t + "\n")
    print(t)


if __name__ == "__main__":
    main()
