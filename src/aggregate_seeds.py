#!/usr/bin/env python3
"""Aggregate multi-seed GloVe eval runs: mean +/- std per condition, and a
noise-aware paired comparison (e.g. lateral vs matched-width symmetric).

Robust to the results-table quirks seen so far:
  * metric columns are auto-detected as the run of float-parseable tokens after
    N, so trailing non-numeric columns (w, dim, ep) are ignored automatically.
  * the "condition id" is the model name with the seed token stripped
    (seed##/-s##), so rows differing only by seed group together. This keys on
    the EXACT name, sidestepping the d064 vs d064x2 label ambiguity -- two runs
    are the same condition iff their names match apart from the seed.

Usage:
  python aggregate_seeds.py --results runs.txt
  python aggregate_seeds.py --results runs.txt --pair asym~con-w15 sym~sum-d128-w15
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED_RE = re.compile(r"[-_](?:seed|s)\d+")


def strip_seed(name):
    return SEED_RE.sub("", name)


def parse_results(path):
    metric_names = None
    header = None
    groups = defaultdict(lambda: defaultdict(list))   # cond_id -> metric -> [vals]
    meta = {}                                          # cond_id -> (w, dim)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            toks = line.split()
            if not toks:
                continue
            if toks[0].lower() == "model":
                header = toks[2:]
                continue
            if set(toks[0]) <= set("-"):
                continue
            if header is None:
                continue
            name = toks[0]
            # maximal run of floats after N (col 1)
            vals = []
            for t in toks[2:]:
                try:
                    vals.append(float(t))
                except ValueError:
                    break
            if not vals:
                continue
            if metric_names is None:
                metric_names = header[:len(vals)]
            cond = strip_seed(name)
            for mname, v in zip(metric_names, vals):
                groups[cond][mname].append(v)
            w = re.search(r"[-_]w(\d+)", name)
            d = re.search(r"[-_]d(\w+?)(?:-|$)", name)   # d064, d064x2, d128
            meta[cond] = (w.group(1) if w else "", d.group(1) if d else "")
    if metric_names is None:
        raise SystemExit("no numeric rows parsed")
    return metric_names, groups, meta


def summarize(groups, metric_names):
    summary = {}
    for cond, mp in groups.items():
        n = max(len(v) for v in mp.values())
        summary[cond] = {m: (float(np.mean(mp[m])), float(np.std(mp[m], ddof=0)),
                             len(mp[m])) for m in metric_names}
        summary[cond]["_n"] = n
    return summary


def print_summary(summary, meta, metric_names):
    conds = sorted(summary, key=lambda c: (meta.get(c, ("", ""))[0], c))
    for cond in conds:
        n = summary[cond]["_n"]
        w, d = meta.get(cond, ("", ""))
        note = "" if n > 1 else "   [!] single seed"
        print(f"\n{cond}   (w={w} dim={d}, seeds={n}){note}")
        for m in metric_names:
            mean, std, k = summary[cond][m]
            print(f"    {m:<8} {mean:.3f} +/- {std:.3f}")


def compare(summary, metric_names, a_key, b_key, sigma):
    def find(key):
        hits = [c for c in summary if key in c]
        if len(hits) != 1:
            raise SystemExit(f"--pair term '{key}' matched {len(hits)} conditions: {hits}")
        return hits[0]
    a, b = find(a_key), find(b_key)
    print(f"\n=== paired comparison: A={a}\n                       B={b} ===")
    print(f"    (A-B; flag = |delta| exceeds {sigma}x pooled seed std)")
    for m in metric_names:
        ma, sa, na = summary[a][m]
        mb, sb, nb = summary[b][m]
        delta = ma - mb
        pooled = float(np.sqrt(sa**2 + sb**2))
        flag = "  *" if abs(delta) > sigma * pooled and pooled > 0 else ""
        if pooled == 0:
            flag = "  (no spread: add seeds)"
        print(f"    {m:<8} A={ma:.3f} B={mb:.3f}  delta={delta:+.3f}  "
              f"pooled_sd={pooled:.3f}{flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"), default=None,
                    help="two substrings each identifying one condition to compare")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--out", default="seed_summary.csv")
    args = ap.parse_args()

    metric_names, groups, meta = parse_results(args.results)
    summary = summarize(groups, metric_names)
    print_summary(summary, meta, metric_names)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "w", "dim", "seeds"]
                   + [f"{m}_mean" for m in metric_names]
                   + [f"{m}_std" for m in metric_names])
        for cond in sorted(summary):
            ww, dd = meta.get(cond, ("", ""))
            row = [cond, ww, dd, summary[cond]["_n"]]
            row += [f"{summary[cond][m][0]:.4f}" for m in metric_names]
            row += [f"{summary[cond][m][1]:.4f}" for m in metric_names]
            w.writerow(row)
    print(f"\nwrote {args.out}")

    if args.pair:
        compare(summary, metric_names, args.pair[0], args.pair[1], args.sigma)


if __name__ == "__main__":
    main()
