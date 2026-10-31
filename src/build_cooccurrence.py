#!/usr/bin/env python3
"""Build a GloVe co-occurrence matrix with route-B (content-budgeted) dynamic windowing.

The window's span varies per occurrence. Frequent "transparent" tokens do not
consume window budget, so the reach stretches *through* function words and
punctuation to collect L *content* tokens per side. That makes X_ij depend on
what actually sat between i and j in each occurrence -- the context-dependence
that separates route B from a fixed (word, distance) kernel (route A in disguise).

--no-dynamic-window forces the transparent set empty, which collapses the whole
thing to classic GloVe: a fixed physical window with harmonic 1/d weights.
So the flag is the clean A/B -- one code path, transparency on vs off.

Transparency is frequency-based here (ports straight from the pp threshold, a
corpus statistic). To switch the criterion to POS or boundary conditioning,
replace build_transparent_set() ONLY -- the windowing machinery below is
criterion-agnostic in structure.

--------------------------------------------------------------------------------
TWO INDEPENDENT WINDOW KNOBS (this is the fix for the pp/reach confound)
--------------------------------------------------------------------------------
  --window L      content budget: stop a side after L *content* tokens.
  --max-span S    physical cap:   stop a side after S *positions*, regardless.
A side stops at whichever binds first (L, S, or sentence end).

With a bare content budget, physical reach silently drifts across the pp sweep:
at low pp many tokens are transparent, so reaching L content tokens travels much
further physically than at high pp. That confounds "which tokens count" with
"how far the window reaches". --max-span pins reach so pp and reach can be swept
as two independent axes.

ROUTE (a) -- data-derived FIXED window (measure -> pick a summary -> hold it):
  1. python build_cooccurrence.py --measure-spans --corpus X.txt \
         --transparent-percentile PP --window L
     -> prints max / mean / median (and percentiles) of the per-occurrence
        physical span needed to collect L content tokens at that pp.
  2. Rebuild with the chosen summary as a fixed physical window (budget lifted
     so the physical cap is the sole reach constraint; frequency filtering still
     governs *weighting*, which you said works):
     python build_cooccurrence.py --corpus X.txt --transparent-percentile PP \
         --window 100000 --max-span <max|mean|median>

"Padding for shorter windows" is a no-op in the count formulation: an occurrence
that reaches the sentence end (or the cap) before filling the window simply
contributes fewer increments -- already correct, nothing to pad. No <PAD> token.

Outputs (--out-dir): cooccurrence.npz (symmetric CSR), vocab.txt, transparent.txt,
meta.json.  With --measure-spans: prints a report and writes spans.json only.
"""
import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse


# --- token predicates -------------------------------------------------------

def is_punct(tok):
    """Pure-punctuation token (Unicode category P*), multilingual-safe."""
    return tok != "" and all(unicodedata.category(ch).startswith("P") for ch in tok)


def build_transparent_set(id2freq, percentile, abs_threshold, dynamic_on):
    """Return (transparent_id_set, cutoff) -- ids treated as skippable (no budget).

    *** This is the single swap point. *** For a POS- or boundary-based criterion,
    rewrite this function to return whichever ids should be transparent; nothing
    downstream cares how the set was chosen.

    dynamic_on=False -> empty set -> classic GloVe fixed physical window.
    abs_threshold (freq floor) overrides the percentile route when given.
    """
    if not dynamic_on:
        return set(), None
    freqs = np.array([id2freq[i] for i in range(len(id2freq))], dtype=np.float64)
    if abs_threshold is not None:
        cutoff = float(abs_threshold)
    else:
        cutoff = float(np.percentile(freqs, percentile))
    return {i for i in range(len(freqs)) if freqs[i] >= cutoff}, cutoff


# --- corpus reading & vocab -------------------------------------------------

def iter_sentences(path, max_sentences=None):
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            if max_sentences is not None and n >= max_sentences:
                break
            toks = line.split()
            if toks:
                yield toks


def build_vocab(path, min_count, strip_punct, max_sentences):
    raw = Counter()
    for toks in iter_sentences(path, max_sentences):
        raw.update(toks)
    kept = [
        (w, c) for w, c in raw.items()
        if c >= min_count and not (strip_punct and is_punct(w))
    ]
    kept.sort(key=lambda wc: (-wc[1], wc[0]))          # freq desc, stable
    word2id = {w: i for i, (w, _) in enumerate(kept)}
    id2freq = {i: c for i, (_, c) in enumerate(kept)}
    return word2id, id2freq, [w for w, _ in kept]


# --- span measurement (route-a helper) --------------------------------------

def measure_spans(path, word2id, transparent, L, max_sentences):
    """Physical span (right side) needed to collect L content tokens, per center.

    Mirrors the counting scan (right-only; the left side is symmetric). Records
    the span only for centers that actually reach L content tokens before the
    sentence ends -- boundary-truncated centers would bias the summary downward,
    so they are counted separately and reported as a truncation rate.
    """
    spans = []
    n_trunc = 0
    for toks in iter_sentences(path, max_sentences):
        ids = [word2id[t] for t in toks if t in word2id]
        n = len(ids)
        for p in range(n):
            k = 0
            span = None
            for q in range(p + 1, n):
                if ids[q] not in transparent:
                    k += 1
                    if k >= L:
                        span = q - p
                        break
            if span is not None:
                spans.append(span)
            else:
                n_trunc += 1
    return np.asarray(spans, dtype=np.float64), n_trunc


def span_report(spans, n_trunc, L, pp):
    n = len(spans)
    total = n + n_trunc
    if n == 0:
        return {"note": "no centers reached the content budget", "budget_L": L,
                "pp": pp, "n_centers": total, "truncated": n_trunc}
    pct = lambda q: float(np.percentile(spans, q))
    rep = {
        "pp": pp, "budget_L": L, "n_centers": total,
        "n_filled": n, "truncated": n_trunc,
        "truncation_rate": round(n_trunc / total, 4),
        "min": float(spans.min()), "mean": round(float(spans.mean()), 3),
        "median": pct(50), "p75": pct(75), "p90": pct(90),
        "p95": pct(95), "p99": pct(99), "max": float(spans.max()),
        "std": round(float(spans.std()), 3),
        # route-(a) window-size candidates:
        "window_max": int(round(spans.max())),
        "window_mean": int(round(spans.mean())),
        "window_median": int(round(pct(50))),
    }
    return rep


# --- co-occurrence ----------------------------------------------------------

def build_cooccurrence(path, word2id, transparent, L, max_span, distance,
                       skip_mode, soft_weight, max_sentences, symmetrize=True):
    """Right-only scan with per-occurrence content budget AND physical cap.

    A side stops when content rank k exceeds L, OR physical distance exceeds
    max_span (None = uncapped), OR the sentence ends. Transparent contexts do
    not spend budget; in hard mode they contribute nothing, in soft mode a small
    soft_weight/phys increment.

    symmetrize=True  -> X = X_R + X_R^T : standard symmetric GloVe co-occurrence.
    symmetrize=False -> X = X_R (asymmetric, right-only): X[i,j] counts j to the
      RIGHT of i, so X^T is the left context. This is the LATERALISED matrix --
      the target vectors W and context vectors W~ then specialise into directional
      roles instead of collapsing. Pair with train_glove.py --combine concat to
      keep the two directions; --combine sum would re-blend them.
    """
    counts = defaultdict(float)
    for toks in iter_sentences(path, max_sentences):
        ids = [word2id[t] for t in toks if t in word2id]   # drop OOV/rare/punct
        n = len(ids)
        for p in range(n):
            center = ids[p]
            k = 1                                           # content rank (1-based)
            q = p + 1
            while q < n and k <= L:
                phys = q - p
                if max_span is not None and phys > max_span:
                    break
                ctx = ids[q]
                if ctx in transparent:
                    if skip_mode == "soft":
                        counts[(center, ctx)] += soft_weight / phys
                    # hard: contribute nothing; either way do not spend budget
                else:
                    w = (1.0 / k) if distance == "content-rank" else (1.0 / phys)
                    counts[(center, ctx)] += w
                    k += 1
                q += 1

    V = len(word2id)
    if counts:
        rows = np.fromiter((i for (i, _) in counts), dtype=np.int32, count=len(counts))
        cols = np.fromiter((j for (_, j) in counts), dtype=np.int32, count=len(counts))
        vals = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
    else:
        rows = cols = np.empty(0, np.int32)
        vals = np.empty(0, np.float64)
    X = sparse.coo_matrix((vals, (rows, cols)), shape=(V, V)).tocsr()
    if symmetrize:
        X = X + X.T                                         # symmetric GloVe
    return X                                                # else asymmetric X_R


# --- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="one sentence per line, ws-tokenised")
    ap.add_argument("--out-dir", default="cooc")
    ap.add_argument("--window", type=int, default=10, help="content budget per side (L)")
    ap.add_argument("--max-span", type=int, default=None,
                    help="physical cap per side (positions); None = uncapped")
    ap.add_argument("--measure-spans", action="store_true",
                    help="report per-occurrence span distribution at this pp, then exit "
                         "(no matrix built)")
    ap.add_argument("--dynamic-window", dest="dynamic", action="store_true", default=True)
    ap.add_argument("--no-dynamic-window", dest="dynamic", action="store_false",
                    help="transparent set empty -> classic GloVe physical window")
    ap.add_argument("--transparent-percentile", type=float, default=99.0,
                    help="freq-value percentile over vocab types; >= cutoff is transparent")
    ap.add_argument("--freq-threshold", type=int, default=None,
                    help="absolute freq floor for transparency (overrides percentile)")
    ap.add_argument("--distance", choices=["content-rank", "physical"], default="content-rank",
                    help="weight metric for content contexts")
    ap.add_argument("--skip-mode", choices=["hard", "soft"], default="hard")
    ap.add_argument("--soft-weight", type=float, default=0.25,
                    help="soft mode: transparent context gets soft_weight/phys")
    ap.add_argument("--strip-punct", dest="strip", action="store_true", default=True)
    ap.add_argument("--no-strip-punct", dest="strip", action="store_false")
    ap.add_argument("--symmetrize", dest="symmetrize", action="store_true", default=True)
    ap.add_argument("--no-symmetrize", dest="symmetrize", action="store_false",
                    help="keep asymmetric right-only X_R (lateralised); train with "
                         "--combine concat")
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--max-sentences", type=int, default=None, help="dev cap")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)                  # persistent-infra fix

    word2id, id2freq, id2word = build_vocab(
        args.corpus, args.min_count, args.strip, args.max_sentences)
    transparent, cutoff = build_transparent_set(
        id2freq, args.transparent_percentile, args.freq_threshold, args.dynamic)

    # --- measurement mode: report spans and exit ---------------------------
    if args.measure_spans:
        spans, n_trunc = measure_spans(
            args.corpus, word2id, transparent, args.window, args.max_sentences)
        rep = span_report(spans, n_trunc, args.window, args.transparent_percentile)
        rep["transparent_cutoff"] = cutoff
        rep["n_transparent"] = len(transparent)
        with open(out / "spans.json", "w") as fh:
            json.dump(rep, fh, indent=2)
        print(f"[measure-spans] pp={args.transparent_percentile} L={args.window} "
              f"transparent={len(transparent)} (cutoff={cutoff})")
        if "note" in rep:
            print("  " + rep["note"]); return
        print(f"  centers={rep['n_centers']} filled={rep['n_filled']} "
              f"truncated={rep['truncated']} ({rep['truncation_rate']:.1%})")
        print(f"  span  min={rep['min']:.0f} mean={rep['mean']:.2f} "
              f"median={rep['median']:.0f} p90={rep['p90']:.0f} "
              f"p95={rep['p95']:.0f} p99={rep['p99']:.0f} max={rep['max']:.0f}")
        print(f"  --max-span candidates:  max={rep['window_max']}  "
              f"mean={rep['window_mean']}  median={rep['window_median']}")
        print(f"  wrote {out / 'spans.json'}")
        return

    # --- build mode --------------------------------------------------------
    X = build_cooccurrence(args.corpus, word2id, transparent, args.window,
                           args.max_span, args.distance, args.skip_mode,
                           args.soft_weight, args.max_sentences, args.symmetrize)
    is_sym = abs(X - X.T).nnz == 0

    sparse.save_npz(out / "cooccurrence.npz", X)
    with open(out / "vocab.txt", "w", encoding="utf-8") as fh:
        for i in range(len(id2word)):
            fh.write(f"{id2word[i]}\t{id2freq[i]}\n")
    with open(out / "transparent.txt", "w", encoding="utf-8") as fh:
        for i in sorted(transparent, key=lambda i: -id2freq[i]):
            fh.write(f"{id2word[i]}\t{id2freq[i]}\n")
    meta = dict(vocab_size=len(id2word), window=args.window, max_span=args.max_span,
                symmetrize=args.symmetrize, is_symmetric=bool(is_sym),
                dynamic=args.dynamic, distance=args.distance, skip_mode=args.skip_mode,
                soft_weight=args.soft_weight, strip_punct=args.strip,
                min_count=args.min_count, transparent_percentile=args.transparent_percentile,
                freq_threshold=args.freq_threshold, transparent_cutoff=cutoff,
                n_transparent=len(transparent), nnz=int(X.nnz),
                x_min=float(X.data.min()) if X.nnz else 0.0,
                x_max=float(X.data.max()) if X.nnz else 0.0)
    with open(out / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print(f"vocab={meta['vocab_size']} transparent={meta['n_transparent']} "
          f"(cutoff={cutoff}) window={args.window} max_span={args.max_span} "
          f"symmetric={meta['is_symmetric']} "
          f"nnz={meta['nnz']} X in [{meta['x_min']:.3f}, {meta['x_max']:.1f}] -> {out}")


if __name__ == "__main__":
    main()
