#!/usr/bin/env python3
"""Generate the two paper figures from results_d128.txt.

Figure 1 (fig_window_30k.pdf): window-size acquisition trends at 30k, d128.
    7 metric panels; x = training window. GloVe (cold-only, it is the warm
    source) across w05-w70; CBOW and SG as cold (solid) and warm30 (dashed),
    available at w05/w10/w15 in the current seeded set.

Figure 2 (fig_size.pdf): corpus-size acquisition trends at d128, w15.
    7 metric panels; x = corpus size {10k, 30k}. Same three architectures,
    cold (solid) and warm30 (dashed); GloVe cold-only.

Encoding (dataviz skill, Okabe-Ito, palette validated):
    GloVe = #0072B2, CBOW = #D55E00, SG = #009E73  (identity = hue)
    cold = solid, warm30 = dashed                   (condition = dash)
Error bars are the per-cell seed std (N=3) from the '±' column.
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "results_d128.txt")
METRICS = ["STS-A", "SICK-A", "SL-999", "WS-353", "Analogy", "Outlier", "OPP"]
MLABEL = {"STS-A": "STS", "SICK-A": "SICK", "SL-999": "SimLex-999",
          "WS-353": "WS-353", "Analogy": "Analogy",
          "Outlier": "Outlier", "OPP": "OPP"}
COL = {"glove": "#0072B2", "cbow": "#D55E00", "cbowNs": "#D55E00", "skgr": "#009E73"}
ALABEL = {"glove": "GloVe", "cbow": "CBOW", "cbowNs": "CBOW", "skgr": "SG"}

VALRE = re.compile(r"(-?\d\.\d{3})±(\d\.\d{3})")
NAMERE = re.compile(
    r"sen-r(\d+k)-(glove|cbowNs|cbow|skgr)-(cold|plain|warm\d+)-d(\d+).*?-w(\d+)")


def parse(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("Model"):
                continue
            nm = NAMERE.search(line)
            if not nm:
                continue
            corpus, arch, cond, dim, win = nm.groups()
            corpus = f"{int(corpus[:-1])}k"   # r030k -> 30k, r010k -> 10k
            vals = VALRE.findall(line)
            if len(vals) != len(METRICS):
                continue
            rec = {"corpus": corpus, "arch": arch, "cond": cond,
                   "dim": int(dim), "window": int(win)}
            for m, (mu, sd) in zip(METRICS, vals):
                rec[m] = (float(mu), float(sd))
            rows.append(rec)
    return rows


def series(rows, corpus, arch, cond, metric, by="window", fixed=None):
    sel = [r for r in rows if r["corpus"] == corpus and r["arch"] == arch
           and r["cond"] == cond]
    if fixed:
        for k, v in fixed.items():
            sel = [r for r in sel if r[k] == v]
    sel.sort(key=lambda r: r[by])
    xs = [r[by] for r in sel]
    mu = [r[metric][0] for r in sel]
    sd = [r[metric][1] for r in sel]
    return xs, np.array(mu), np.array(sd)


def style(ax):
    ax.grid(True, color="#E6E6E6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#B0B0B0")
    ax.tick_params(colors="#606060", labelsize=8)


def plot_line(ax, xs, mu, sd, color, dashed, marker="o"):
    if len(xs) == 0:
        return
    ax.errorbar(xs, mu, yerr=sd, color=color,
                ls="--" if dashed else "-", lw=2.0 if not dashed else 1.8,
                marker=marker, ms=5, mew=0, capsize=2, elinewidth=1,
                ecolor=color, alpha=0.95, zorder=3)


def legend_handles():
    return [
        Line2D([], [], color=COL["glove"], lw=2, marker="o", ms=5, label="GloVe (source)"),
        Line2D([], [], color=COL["cbow"], lw=2, marker="o", ms=5, label="CBOW"),
        Line2D([], [], color=COL["skgr"], lw=2, marker="o", ms=5, label="SG"),
        Line2D([], [], color="#606060", lw=2, ls="-", label="cold start"),
        Line2D([], [], color="#606060", lw=1.8, ls="--", label="warm start (GloVe w30)"),
    ]


# Exact canvas so the paper's \includegraphics*[bb=0 0 780 380] matches the PDF
# MediaBox precisely (780/72 x 380/72 in) and clips nothing.
CANVAS = (780 / 72.0, 380 / 72.0)


def panel_grid(title):
    fig, axes = plt.subplots(2, 4, figsize=CANVAS)
    fig.suptitle(title, fontsize=12, y=0.985)
    axes = axes.ravel()
    return fig, axes


# ----------------------------------------------------------------------
def figure_window(rows):
    fig, axes = panel_grid(
        "Window-size acquisition at 30k articles (d=128)")
    for k, metric in enumerate(METRICS):
        ax = axes[k]
        style(ax)
        # GloVe cold-only, full window range
        plot_line(ax, *series(rows, "30k", "glove", "plain", metric),
                  COL["glove"], dashed=False)
        # CBOW cold / warm30
        plot_line(ax, *series(rows, "30k", "cbow", "cold", metric),
                  COL["cbow"], dashed=False)
        plot_line(ax, *series(rows, "30k", "cbow", "warm30", metric),
                  COL["cbow"], dashed=True)
        # SG cold / warm30
        plot_line(ax, *series(rows, "30k", "skgr", "cold", metric),
                  COL["skgr"], dashed=False)
        plot_line(ax, *series(rows, "30k", "skgr", "warm30", metric),
                  COL["skgr"], dashed=True)
        ax.set_title(MLABEL[metric], fontsize=10, color="#303030")
        ax.set_xlabel("window", fontsize=8)
    axes[-1].axis("off")
    axes[-1].legend(handles=legend_handles(), loc="center", frameon=False,
                    fontsize=9, title="", handlelength=2.4)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("fig_window_30k.pdf")
    plt.close(fig)
    print("wrote fig_window_30k.pdf")


def figure_size(rows):
    fig, axes = panel_grid(
        "Corpus-size acquisition (d=128, window=15)")
    sizes = ["10k", "30k"]
    xpos = {"10k": 10, "30k": 30}
    for k, metric in enumerate(METRICS):
        ax = axes[k]
        style(ax)
        for arch, cond, dashed in (("glove", "plain", False),
                                   ("cbow", "cold", False),
                                   ("cbow", "warm30", True),
                                   ("skgr", "cold", False),
                                   ("skgr", "warm30", True)):
            xs, mu, sd = [], [], []
            for sz in sizes:
                s = series(rows, sz, arch, cond, metric,
                           fixed={"window": 15})
                if len(s[0]):
                    xs.append(xpos[sz]); mu.append(s[1][0]); sd.append(s[2][0])
            plot_line(ax, xs, np.array(mu), np.array(sd),
                      COL[arch], dashed=dashed)
        ax.set_title(MLABEL[metric], fontsize=10, color="#303030")
        ax.set_xticks([10, 30]); ax.set_xticklabels(["10k", "30k"])
        ax.set_xlim(5, 35)
        ax.set_xlabel("corpus", fontsize=8)
    axes[-1].axis("off")
    axes[-1].legend(handles=legend_handles(), loc="center", frameon=False,
                    fontsize=9, handlelength=2.4)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("fig_size.pdf")
    plt.close(fig)
    print("wrote fig_size.pdf")


if __name__ == "__main__":
    rows = parse(SRC)
    print(f"parsed {len(rows)} rows")
    figure_window(rows)
    figure_size(rows)
