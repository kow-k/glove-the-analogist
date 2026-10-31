#!/usr/bin/env python3
"""Implementation-independent computational-cost comparison of the three
training conditions: Skip-gram-only vs GloVe→CBOW sequence (+ CBOW-cold ref).

Wall-clock conflates algorithm with implementation substrate (gensim Cython vs
PyTorch) and with the softmax choice. This instead counts *intrinsic work* in
substrate-free units, so the comparison reflects the algorithms, not the code.

UNITS. Every per-event core operation is a d-dimensional dot/update (a dot
product or a vector add/scale over the embedding dimension d). We count the
NUMBER of such d-ops; the absolute FLOP count is ≈ 2·d·(that number). Since all
three methods scale linearly in d, d cancels in every ratio — so the ratios are
dimension-independent and are the headline.

COST MODEL (leading terms; lower-order terms noted, bias scalars ignored):

  Let  T   = # in-vocab training tokens (targets),
       V   = vocabulary size,
       NNZ = # stored nonzero cells in the co-occurrence matrix,
       w_x = one-sided window, E_x = epochs, k_x = negatives (per method x).

  Skip-gram (SGNS):      each target pairs with ~2·w·s context words, each pair
                         doing 1 positive + k negative d-dot updates.
      cost_SG   = E_sg · T · (2·w_sg·s) · (1 + k_sg)
      (s = gensim dynamic-window shrink ≈ 0.5: the window is sampled uniformly
       up to w each time. Set --gensim-shrink 1.0 to disable.)

  CBOW, full softmax (your current trainer): one averaged context per target,
                         then an output projection over the whole vocabulary.
      cost_CBOW_sm = E_cbow · T · ( 2·w_cbow   +   V )
                                   └ pooling ┘   └ softmax (dominant) ┘

  CBOW, negative sampling (hypothetical, for the fair "is it cheap?" question):
      cost_CBOW_ns = E_cbow · T · ( 2·w_cbow   +   (1 + k_cbow) )

  GloVe (factorization): one d-dot per nonzero cell per epoch.
      cost_GloVe = E_glove · NNZ
      (co-occurrence BUILD is dimension-independent ≈ T·2·w counting increments;
       reported separately, excluded from the d-op totals.)

  Sequence = GloVe + CBOW (warm-init copy is a one-off O(V) load, negligible).

READINGS the script prints:
  * cost_SG, cost_sequence (softmax CBOW = as implemented), cost_sequence (ns),
    cost_CBOW alone (both variants), and ratios vs SG.
  * The point it makes: with a FULL-SOFTMAX CBOW the sequence is dominated by
    T·V and is far more expensive than SG; the "sequence is cheap" intuition
    only holds if CBOW uses negative sampling, where its single-context-per-
    target structure makes it cheaper per epoch than SG.

Inputs (read from artifacts, or pass numbers directly):
  --vocab vocab.txt | --vocab-size V
  --corpus file     | --tokens T     (corpus tokens counted after min-count via --vocab)
  --cooc cooccurrence.npz | --nnz NNZ
"""
import argparse, sys
from pathlib import Path


def count_tokens(corpus, vocab_set, min_count):
    """# training tokens = corpus tokens present in vocab (targets)."""
    n = 0
    with open(corpus, encoding="utf-8", errors="replace") as f:
        for line in f:
            for t in line.split():
                if vocab_set is None or t in vocab_set:
                    n += 1
    return n


def load_vocab(path):
    toks = [ln.split()[0] for ln in open(path, encoding="utf-8") if ln.strip()]
    # a word2vec header line "N D" would sneak in as two ints; drop if so
    if len(toks) >= 1 and toks[0].isdigit():
        toks = toks[1:]
    return toks


def load_nnz(npz):
    try:
        from scipy.sparse import load_npz
        return int(load_npz(npz).nnz)
    except Exception:
        import numpy as np
        z = np.load(npz, allow_pickle=True)
        for k in ("data", "nnz"):
            if k in z:
                return int(z[k].shape[0]) if k == "data" else int(z[k])
        raise SystemExit(f"could not read nnz from {npz}; pass --nnz")


def fmt(x):
    for u, s in [(1e12,"T"),(1e9,"G"),(1e6,"M"),(1e3,"k")]:
        if abs(x) >= u:
            return f"{x/u:.2f}{s}"
    return f"{x:.0f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab"); ap.add_argument("--vocab-size", type=int)
    ap.add_argument("--corpus"); ap.add_argument("--tokens", type=int)
    ap.add_argument("--cooc"); ap.add_argument("--nnz", type=int)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--sg-window", type=int, default=15)
    ap.add_argument("--sg-epochs", type=int, default=5)
    ap.add_argument("--sg-negatives", type=int, default=5)
    ap.add_argument("--gensim-shrink", type=float, default=0.5,
                    help="gensim dynamic-window average factor (~0.5); 1.0 disables")
    ap.add_argument("--cbow-window", type=int, default=10)
    ap.add_argument("--cbow-epochs", type=int, default=7)
    ap.add_argument("--cbow-negatives", type=int, default=5,
                    help="k for the hypothetical negative-sampling CBOW")
    ap.add_argument("--glove-epochs", type=int, default=20)
    a = ap.parse_args()

    # resolve V, T, NNZ
    vocab_set = None
    if a.vocab_size:
        V = a.vocab_size
    elif a.vocab:
        toks = load_vocab(a.vocab); V = len(toks); vocab_set = set(toks)
    else:
        sys.exit("need --vocab or --vocab-size")
    if a.tokens:
        T = a.tokens
    elif a.corpus:
        T = count_tokens(a.corpus, vocab_set, a.min_count)
    else:
        sys.exit("need --corpus or --tokens")
    if a.nnz:
        NNZ = a.nnz
    elif a.cooc:
        NNZ = load_nnz(a.cooc)
    else:
        sys.exit("need --cooc or --nnz")

    s = a.gensim_shrink
    # costs in d-ops (divide FLOPs by ~2d)
    sg        = a.sg_epochs   * T * (2*a.sg_window*s) * (1 + a.sg_negatives)
    cbow_sm   = a.cbow_epochs * T * (2*a.cbow_window + V)
    cbow_ns   = a.cbow_epochs * T * (2*a.cbow_window + 1 + a.cbow_negatives)
    glove     = a.glove_epochs * NNZ
    seq_sm    = glove + cbow_sm
    seq_ns    = glove + cbow_ns

    print("="*74)
    print("Computational cost (intrinsic d-ops; d cancels in all ratios)")
    print(f"  T(tokens)={fmt(T)}  V(vocab)={fmt(V)}  NNZ={fmt(NNZ)}")
    print(f"  SG(w{a.sg_window},ep{a.sg_epochs},k{a.sg_negatives},shrink{s})  "
          f"CBOW(w{a.cbow_window},ep{a.cbow_epochs},k{a.cbow_negatives})  "
          f"GloVe(ep{a.glove_epochs})")
    print("="*74)
    rows = [
        ("Skip-gram only (SGNS)",            sg),
        ("GloVe stage",                      glove),
        ("CBOW stage — full softmax (now)",  cbow_sm),
        ("CBOW stage — negative sampling",   cbow_ns),
        ("SEQUENCE = GloVe+CBOW (softmax)",  seq_sm),
        ("SEQUENCE = GloVe+CBOW (neg-samp)", seq_ns),
    ]
    for name, c in rows:
        print(f"  {name:36s} {fmt(c):>9}   {c/sg:6.2f}× SG")
    print("-"*74)
    print("Reading:")
    print(f"  • Full-softmax CBOW costs ≈ T·V = {fmt(T)}·{fmt(V)} per epoch; the")
    print(f"    V-term dominates, so the sequence is {seq_sm/sg:.0f}× SG — an O(V)")
    print(f"    softmax effect, NOT a gensim-vs-PyTorch effect.")
    print(f"  • With negative sampling the CBOW stage is {cbow_ns/sg:.2f}× SG")
    print(f"    (one averaged context per target vs SG's ~2w contexts), and the")
    print(f"    whole sequence is {seq_ns/sg:.2f}× SG — i.e. the 'cheap' claim holds")
    print(f"    only in this regime, and only if NS-CBOW matches softmax quality.")
    print("\nModel caveats: leading-order; ignores bias scalars, subsampling (shrinks")
    print("T for SG), and sentence-edge window truncation; gensim shrink≈0.5 applied")
    print("to SG. Build cost (dimension-independent, ≈T·2w increments) excluded.")


if __name__ == "__main__":
    main()
