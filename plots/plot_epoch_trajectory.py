#!/usr/bin/env python3
"""CBOW epoch-trajectory plot: metrics vs training epoch.

Reads the per-epoch checkpoint JSONs produced by
  train_cbow_ns.py --save-each-epoch ...   (then evaluated with the harness)
i.e. files named ...-ep01-r<seed>.json ... -epNN-r<seed>.json, and plots each
metric against epoch. Averages over seeds when several are present (mean +- s.d.).

Two panels (no dual axis): composition (STS-A, SICK-A) and relational
(Analogy, WS-353, SimLex-999, OPP) — the two groups that trade off.

Usage:
  python plot_epoch_trajectory.py --dir models-100k-traj \
      --glob 'sen-r100k-cbowNs-cold-d128-w10-*ep*-r*.json' --out fig_epoch_w10.pdf
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
COMPOSITION = ["STS-A", "SICK-A"]
RELATIONAL  = ["Analogy", "WS-353", "SL-999", "OPP"]
# Okabe-Ito (validated): one hue per metric line
COL = {"STS-A": "#D55E00", "SICK-A": "#E69F00",
       "Analogy": "#0072B2", "WS-353": "#009E73",
       "SL-999": "#56B4E9", "OPP": "#CC79A7"}

EP_RE = re.compile(r"-ep(\d+)(?:-r\d+)?(?:\.json)?$")


def metric(d, label):
    for k in METRIC_KEYS[label]:
        if isinstance(d.get(k), (int, float)):
            return float(d[k])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--glob", default="*cbowNs*ep*.json")
    ap.add_argument("--out", default="fig_epoch_trajectory.pdf")
    ap.add_argument("--title", default="CBOW epoch trajectory (d=128)")
    ap.add_argument("--band", choices=["std", "sem", "none"], default="std",
                    help="shaded interval across seeds: std (default), sem, or none")
    a = ap.parse_args()

    # epoch -> metric -> list of per-seed values
    by_ep = defaultdict(lambda: defaultdict(list))
    n = 0
    for p in glob.glob(os.path.join(a.dir, a.glob)):
        m = EP_RE.search(os.path.basename(p)[:-5] if p.endswith(".json") else p)
        if not m:
            continue
        ep = int(m.group(1))
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"  [skip] {os.path.basename(p)}: {e}"); continue
        for lab in METRIC_KEYS:
            v = metric(d, lab)
            if v is not None:
                by_ep[ep][lab].append(v)
        n += 1
    if not by_ep:
        raise SystemExit(f"no epoch JSONs matched in {a.dir} (check --glob)")
    eps = sorted(by_ep)
    print(f"parsed {n} files; epochs {eps}")

    def series(lab):
        xs, mu, sd, ns = [], [], [], []
        for e in eps:
            vals = by_ep[e].get(lab, [])
            if vals:
                xs.append(e); mu.append(np.mean(vals))
                sd.append(np.std(vals)); ns.append(len(vals))
        return xs, np.array(mu), np.array(sd), np.array(ns)

    def band_hw(sd, ns):
        if a.band == "none":
            return np.zeros_like(sd)
        if a.band == "sem":
            return sd / np.sqrt(np.maximum(ns, 1))
        return sd

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.suptitle(a.title, fontsize=12, y=0.99)
    for ax, group, gname in ((axes[0], COMPOSITION, "Composition"),
                             (axes[1], RELATIONAL, "Relational / analogical")):
        ax.grid(True, color="#E6E6E6", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color("#B0B0B0")
        ax.tick_params(colors="#606060", labelsize=9)
        for lab in group:
            xs, mu, sd, ns = series(lab)
            if len(xs):
                ax.plot(xs, mu, color=COL[lab], lw=2, marker="o", ms=5, mew=0,
                        label=lab)
                hw = band_hw(sd, ns)
                if a.band != "none" and len(xs) > 1 and np.any(hw > 0):
                    ax.fill_between(xs, mu - hw, mu + hw, color=COL[lab],
                                    alpha=0.18, lw=0)
        ax.set_title(gname, fontsize=11, color="#303030")
        ax.set_xlabel("training epoch", fontsize=9)
        ax.set_xticks(eps)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
