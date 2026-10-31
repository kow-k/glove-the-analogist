#!/usr/bin/env python3
"""Cross-METHOD CBOW-vs-SG(-vs-GloVe) epoch trajectory.

The discriminator: does Skip-gram also peak-then-erode on the relational metrics
like CBOW, or does it converge monotonically? Overlays each method's per-epoch
trajectory for a few diagnostic metrics (default Analogy and STS-A), one line per
method.

Each positional argument is  LABEL=DIR  or  LABEL=DIR=GLOB  (give a per-method
glob when methods share a directory but differ by filename, e.g. cbowNs vs skgr):

  python plot_epoch_methods.py \
    "CBOW=models-traj=*cbowNs-cold-d128-w10-*ep*-r*.json" \
    "SG=models-traj=*skgr-cold-d128-w10-*ep*-r*.json" \
    --metrics Analogy STS-A WS-353 --out fig_epoch_methods_w10.pdf

Seeds are averaged (mean +- s.d.). Note the epoch axis is only comparable across
methods at the SAME corpus (one epoch = one pass = same token count for both).
"""
import argparse, glob, json, os, re
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRIC_KEYS = {
    "STS-A":   ["sts_plain", "sts_a", "sts"],
    "SICK-A":  ["sick_plain", "sick_a", "sick"],
    "SL-999":  ["simlex", "sl999"],
    "WS-353":  ["wordsim", "ws353"],
    "Analogy": ["analogy"],
    "OPP":     ["outlier_opp", "opp"],
    "Outlier": ["outlier"],
}
_CANON = {k.lower(): k for k in METRIC_KEYS}   # resolve requested labels case-insensitively
# one hue per method (Okabe-Ito, validated); extra labels fall back to a cycle
METHOD_COL = {"CBOW": "#D55E00", "SG": "#009E73", "GloVe": "#0072B2",
              "PPMI-SVD": "#CC79A7"}
FALLBACK = ["#000000", "#E69F00", "#56B4E9", "#F0E442"]
EP_RE = re.compile(r"-ep(\d+)(?:-r\d+)?(?:\.json)?$")
STEP_RE = re.compile(r"-st(\d+)(?:-r\d+)?(?:\.json)?$")


def metric(d, label):
    for k in METRIC_KEYS[label]:
        if isinstance(d.get(k), (int, float)):
            return float(d[k])
    return None


def load(dirpath, globpat, xre, mult):
    by_x = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(os.path.join(dirpath, globpat)):
        m = xre.search(os.path.basename(p)[:-5] if p.endswith(".json") else p)
        if not m:
            continue
        x = int(m.group(1)) * mult
        try:
            d = json.load(open(p))
        except Exception:
            continue
        for lab in METRIC_KEYS:
            v = metric(d, lab)
            if v is not None:
                by_x[x][lab].append(v)
    return by_x


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("methods", nargs="+", help="LABEL=DIR or LABEL=DIR=GLOB")
    ap.add_argument("--metrics", nargs="+", default=["Analogy", "STS-A"])
    ap.add_argument("--band", choices=["std", "sem", "none"], default="std",
                    help="shaded interval across seeds: std (default), sem, or none")
    ap.add_argument("--xaxis", choices=["epoch", "step"], default="epoch",
                    help="epoch (per-epoch JSONs) or step (per-update JSONs, "
                         "...-st<N>...). The step axis is corpus/lr-invariant.")
    ap.add_argument("--batch", type=int, default=0,
                    help="with --xaxis step: multiply steps by this batch size to "
                         "show tokens-seen (0 = raw update count)")
    ap.add_argument("--ncols", type=int, default=0,
                    help="panels per row (default 0 = all metrics in one row)")
    ap.add_argument("--out", default="fig_epoch_methods.pdf")
    ap.add_argument("--title", default="Training trajectory by method (d=128)")
    a = ap.parse_args()
    a.metrics = [_CANON.get(m.lower(), m) for m in a.metrics]  # "outlier"->"Outlier", etc.

    xre = STEP_RE if a.xaxis == "step" else EP_RE
    mult = a.batch if (a.xaxis == "step" and a.batch > 0) else 1
    xlabel = ("training epoch" if a.xaxis == "epoch"
              else "tokens seen" if mult != 1 else "optimizer updates")
    default_glob = "*st*-r*.json" if a.xaxis == "step" else "*ep*-r*.json"

    series = []
    for i, item in enumerate(a.methods):
        parts = item.split("=")
        if len(parts) == 2:
            lab, d, gp = parts[0], parts[1], default_glob
        elif len(parts) >= 3:
            lab, d, gp = parts[0], parts[1], "=".join(parts[2:])
        else:
            raise SystemExit(f"expected LABEL=DIR[=GLOB], got {item!r}")
        be = load(d, gp, xre, mult)
        col = METHOD_COL.get(lab, FALLBACK[i % len(FALLBACK)])
        series.append((lab, be, col))
        print(f"{lab}: x-values {sorted(be)[:8]}{' ...' if len(be) > 8 else ''} "
              f"(dir={d}, glob={gp})")

    def pts(be, met):
        eps = sorted(be)
        xs, mu, sd, ns = [], [], [], []
        for e in eps:
            vals = be[e].get(met, [])
            if vals:
                xs.append(e); mu.append(np.mean(vals))
                sd.append(np.std(vals)); ns.append(len(vals))
        return xs, np.array(mu), np.array(sd), np.array(ns)

    def band_hw(sd, ns):
        if a.band == "none":
            return np.zeros_like(sd)
        if a.band == "sem":
            return sd / np.sqrt(np.maximum(ns, 1))
        return sd  # std

    nmet = len(a.metrics)
    ncol = a.ncols if a.ncols > 0 else nmet
    nrow = -(-nmet // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.2 * nrow), squeeze=False)
    for k in range(nmet, nrow * ncol):          # blank any unused trailing panels
        axes.flat[k].set_visible(False)
    fig.suptitle(a.title, fontsize=12, y=0.99)
    for j, met in enumerate(a.metrics):
        ax = axes.flat[j]
        ax.grid(True, color="#E6E6E6", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color("#B0B0B0")
        ax.tick_params(colors="#606060", labelsize=9)
        allx = set()
        for lab, be, col in series:
            xs, mu, sd, ns = pts(be, met)
            if len(xs):
                allx |= set(xs)
                ax.plot(xs, mu, color=col, lw=2, marker="o", ms=5, mew=0, label=lab)
                hw = band_hw(sd, ns)
                if a.band != "none" and len(xs) > 1 and np.any(hw > 0):
                    ax.fill_between(xs, mu - hw, mu + hw, color=col,
                                    alpha=0.18, lw=0)
        ax.set_title(met, fontsize=11, color="#303030")
        ax.set_xlabel(xlabel, fontsize=9)
        if allx and a.xaxis == "epoch":
            ax.set_xticks(sorted(allx))      # steps are dense -> let matplotlib tick
        ax.legend(frameon=False, fontsize=9, title="method")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
