#!/usr/bin/env python3
"""Block D warm-vs-cold transfer analysis.

Tests the general transfer thesis: a warm-start source hands a CBOW its OWN
signature capability. Reads the per-seed evaluation JSONs the campaign writes
(cbowNs-cold, cbowNs-warm<src>, cbowNs-warmPS<src>), matches each warm run to the
cold baseline at the same seed and training window, and for each warm ARM
(GloVe-sourced, PPMI-SVD-sourced) reports per metric:
    n+ / n- (warm>cold / warm<cold over the source x seed grid),
    mean warm-cold delta, and a two-sided sign-test p-value.

Prediction (flagged with * = predicted to lift):
    GloVe-warm     -> Analogy
    PPMI-SVD-warm  -> WS-353, Outlier, OPP, SimLex-999
Neither should lift composition (STS/SICK), which is CBOW's own.

Usage:
    python analyze_warm_transfer.py --dir models-100k
    python analyze_warm_transfer.py --dir models-100k --train-window 7 --by-source
"""
import argparse, glob, json, os, re
from collections import defaultdict

# metric label -> candidate JSON keys (first present wins). Override if your
# batch-evaluate-to-json.sh uses different names.
METRIC_KEYS = {
    "STS-A":   ["sts_plain", "sts_a", "sts"],
    "SICK-A":  ["sick_plain", "sick_a", "sick"],
    "SL-999":  ["simlex", "sl999", "sl_999"],
    "WS-353":  ["wordsim", "ws353", "ws_353"],
    "Analogy": ["analogy", "analogy_acc"],
    "Outlier": ["outlier", "outlier_acc"],
    "OPP":     ["outlier_opp", "opp"],
}
METRICS = list(METRIC_KEYS)
PREDICTED = {"GloVe": {"Analogy"},
             "PPMI-SVD": {"WS-353", "Outlier", "OPP", "SL-999"}}

# cond: cold | warm<N> (GloVe source) | warmPS<N> (PPMI-SVD source)
COND_RE = re.compile(r"-(cold|warmPS(\d+)|warm(\d+))-")
DW_RE   = re.compile(r"-d(\d+)-w(\d+)-")
LR_RE   = re.compile(r"-lr([0-9.]+)-")
EP_RE   = re.compile(r"-ep([0-9]+)-")
SEED_RE = re.compile(r"-r(\d+)(?:\.json)?$")


def get_metric(d, label):
    for k in METRIC_KEYS[label]:
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    return None


def parse_name(path):
    name = os.path.basename(path)
    if "cbowns" not in name.lower():
        return None
    mc = COND_RE.search(name)
    mdw = DW_RE.search(name)
    ms = SEED_RE.search(name[:-5] if name.endswith(".json") else name)
    if not (mc and mdw):
        return None
    if mc.group(1) == "cold":
        arm, src = "cold", None
    elif mc.group(2) is not None:          # warmPS<N>
        arm, src = "PPMI-SVD", int(mc.group(2))
    else:                                   # warm<N>
        arm, src = "GloVe", int(mc.group(3))
    seed = int(ms.group(1)) if ms else None
    mlr, mep = LR_RE.search(name), EP_RE.search(name)
    return {"arm": arm, "src": src, "dim": int(mdw.group(1)),
            "twin": int(mdw.group(2)),
            "lr": mlr.group(1) if mlr else "?",
            "ep": mep.group(1) if mep else "?",
            "seed": seed, "name": name}


def cfg_key(r):
    """Everything that must match between a warm run and its cold baseline,
    i.e. the full config EXCEPT the warm/cold condition."""
    return (r["dim"], r["twin"], r["lr"], r["ep"], r["seed"])


def sign_p(nplus, nminus):
    n = nplus + nminus
    if n == 0:
        return None
    try:
        from scipy.stats import binomtest
        return binomtest(min(nplus, nminus), n, 0.5, alternative="two-sided").pvalue
    except Exception:
        from math import comb
        k = min(nplus, nminus)
        tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="dir with the Block D *.json files")
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--train-window", type=int, default=None,
                    help="restrict to this CBOW training window (default: all)")
    ap.add_argument("--dim", type=int, default=None,
                    help="restrict to this embedding dim (recommended: 128)")
    ap.add_argument("--by-source", action="store_true",
                    help="also break the sign test down per source window")
    a = ap.parse_args()

    rows = []
    for p in glob.glob(os.path.join(a.dir, a.glob)):
        meta = parse_name(p)
        if not meta:
            continue
        if a.train_window is not None and meta["twin"] != a.train_window:
            continue
        if a.dim is not None and meta["dim"] != a.dim:
            continue
        try:
            with open(p) as fh:
                d = json.load(fh)
        except Exception as e:
            print(f"  [skip] {os.path.basename(p)}: {e}")
            continue
        meta["metrics"] = {m: get_metric(d, m) for m in METRICS}
        meta["mtime"] = os.path.getmtime(p)
        meta["path"] = os.path.basename(p)
        rows.append(meta)

    if not rows:
        raise SystemExit(f"no cbowNs JSONs matched in {a.dir} "
                         f"(check --glob / --train-window / METRIC_KEYS)")

    # ── de-duplicate: a run is identified by (arm, src, full-config). If two
    # files map to the same run (e.g. an old campaign left under a different
    # name/TAG), keep the newest by mtime and report the rest, so a polluted
    # directory can't silently corrupt the baseline.
    groups = defaultdict(list)
    for r in rows:
        groups[(r["arm"], r["src"], cfg_key(r))].append(r)
    deduped, ndrop = [], 0
    for key, grp in groups.items():
        if len(grp) > 1:
            grp.sort(key=lambda r: r["mtime"], reverse=True)
            # do the duplicates disagree on the numbers?
            disagree = any(
                abs((g["metrics"].get(m) or 0) - (grp[0]["metrics"].get(m) or 0)) > 0.003
                for g in grp[1:] for m in METRICS
                if g["metrics"].get(m) is not None and grp[0]["metrics"].get(m) is not None)
            tag = "DIFFERENT numbers" if disagree else "identical"
            print(f"  [dup] {len(grp)} files for {key[0]}"
                  f"{'' if key[1] is None else '/w'+str(key[1])} "
                  f"seed{cfg_key(grp[0])[-1]} ({tag}); keeping newest "
                  f"{grp[0]['path']}, dropping {[g['path'] for g in grp[1:]]}")
            ndrop += len(grp) - 1
        deduped.append(grp[0])
    if ndrop:
        print(f"  [dup] dropped {ndrop} duplicate file(s); "
              f"CLEAN THE DIRECTORY if the duplicates show DIFFERENT numbers "
              f"(two real runs mixed) rather than identical copies.")
    rows = deduped

    # cold baseline indexed by full config (dim, twin, lr, ep, seed)
    cold = {}
    for r in rows:
        if r["arm"] == "cold":
            cold[cfg_key(r)] = r["metrics"]
    dims = sorted({r["dim"] for r in rows})
    twins = sorted({r["twin"] for r in rows})
    print(f"loaded {len(rows)} runs; dims present: {dims}; "
          f"training window(s): {twins}; cold cells: {len(cold)}")
    if len(dims) > 1 and a.dim is None:
        print(f"  [warn] multiple dims present {dims}; warm/cold are matched "
              f"within dim, but pass --dim 128 to be explicit.")
    if not cold:
        raise SystemExit("no cold baseline runs found — need cbowNs-cold-*.json")

    def report(arm, cells):
        # cells: list of warm run dicts for this arm
        matched = [c for c in cells if cfg_key(c) in cold]
        unmatched = len(cells) - len(matched)
        srcs = sorted({c['src'] for c in matched})
        seeds = sorted({c['seed'] for c in matched})
        note = f"; {unmatched} warm runs had no matching cold config (skipped)" \
               if unmatched else ""
        print(f"\n=== {arm}-warm  (sources w{srcs}, seeds {seeds}; "
              f"{len(matched)} matched warm cells{note}) ===")
        print(f"  {'metric':<8} {'n+':>3} {'n-':>3} {'mean Δ':>9} "
              f"{'sign p':>8}   pred")
        for m in METRICS:
            npl = nmi = 0
            dsum = 0.0
            nd = 0
            for c in matched:
                base = cold.get(cfg_key(c))
                if not base:
                    continue
                w, b = c["metrics"].get(m), base.get(m)
                if w is None or b is None:
                    continue
                dz = w - b
                dsum += dz; nd += 1
                if dz > 0: npl += 1
                elif dz < 0: nmi += 1
            if nd == 0:
                print(f"  {m:<8} {'--':>3} {'--':>3} {'  n/a':>9} {'  n/a':>8}")
                continue
            p = sign_p(npl, nmi)
            star = " *" if m in PREDICTED.get(arm, set()) else ""
            pstr = f"{p:.4f}" if p is not None else "  n/a"
            print(f"  {m:<8} {npl:>3} {nmi:>3} {dsum/nd:>+9.4f} {pstr:>8}  {star}")
        if a.by_source:
            for s in srcs:
                sub = [c for c in matched if c["src"] == s]
                print(f"  -- source w{s} --")
                for m in METRICS:
                    ds = [c["metrics"][m] - cold[cfg_key(c)][m]
                          for c in sub
                          if cold.get(cfg_key(c))
                          and c["metrics"].get(m) is not None
                          and cold[cfg_key(c)].get(m) is not None]
                    if ds:
                        npl = sum(1 for x in ds if x > 0)
                        nmi = sum(1 for x in ds if x < 0)
                        print(f"     {m:<8} {npl}+/{nmi}-  meanΔ={sum(ds)/len(ds):+.4f}")

    for arm in ("GloVe", "PPMI-SVD"):
        cells = [r for r in rows if r["arm"] == arm]
        if cells:
            report(arm, cells)
        else:
            print(f"\n=== {arm}-warm: no runs found "
                  f"({'cbowNs-warm*' if arm=='GloVe' else 'cbowNs-warmPS*'}.json) ===")

    print("\n* = metric this source is predicted to lift (its own specialty). "
          "Thesis holds if the starred rows are positive (n+ > n-) and the "
          "unstarred composition rows (STS/SICK) are null.")


if __name__ == "__main__":
    main()
