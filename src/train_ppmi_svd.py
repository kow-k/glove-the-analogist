#!/usr/bin/env python3
"""PPMI-SVD word embeddings (the count-based sibling of GloVe).

Builds (positive) PMI from the SAME word-context co-occurrence matrix GloVe uses,
then truncates it with SVD to `dim` dimensions. This is the canonical count-based
method of Levy, Goldberg & Dagan (2015, TACL), and the closest linear-algebra
cousin to GloVe: both turn co-occurrence statistics into dense vectors by
factorisation rather than prediction.

Drop-in with the project pipeline: reads a --cooc-dir containing
`cooccurrence.npz` (scipy sparse counts X_ij) and `vocab.txt` (one token per
line, optionally "token count", optional word2vec header), exactly like
train_glove.py, and writes a headerless .vec the eval harness reads.
Deterministic (seeded randomized SVD), so N=1 suffices, as for GloVe.

The "joint" / symmetric variant: PMI with context-distribution smoothing
(cds=0.75) is asymmetric, so the left (word, U) and right (context, V) factors
differ; --combine sum outputs W + C, the analogue of GloVe's W + W~. This is the
most likely meaning of "PPMI-SVD joint". --combine word outputs W only.

Recipe knobs (defaults = the Levy et al. recommended strong baseline):
  --shift k    PMI shift log(k): subtract log(neg) like SGNS's k negatives (1=off)
  --cds a      context distribution smoothing exponent (0.75 standard; 1.0 off)
  --sv-power p word vec = U * Sigma**p  (0, 0.5, or 1; 0.5 is Levy's recommended)
  --combine    sum (W+C, "joint") | word (W only) | concat (W,C -> 2*dim)
  --eig-weight alias for --sv-power (compat)

Example (matches the GloVe grid; run once per window on the same cooc dir):
  python train_ppmi_svd.py --cooc-dir cooc-100k/w30 --dim 128 \
      --out models-100k/r100k-ppmisvd-joint-d128-w30.vec
"""
import argparse, os, sys
import numpy as np


def load_vocab(path):
    toks = [ln.split()[0] for ln in open(path, encoding="utf-8") if ln.strip()]
    if len(toks) >= 2 and toks[0].isdigit() and toks[1].isdigit():
        toks = toks[2:]  # drop a stray "N D" word2vec header
    return toks


def load_cooc(npz):
    """Load the word-context co-occurrence matrix, tolerant of how it was saved.

    Tries, in order: scipy.sparse.save_npz format; np.savez CSR
    (data/indices/indptr/shape); np.savez COO (data + row/col, several key
    spellings); a single dense array. If none match, reports the keys found so
    the loader can be matched to build_cooccurrence.py.
    """
    from scipy.sparse import load_npz, csr_matrix, coo_matrix
    # 1) native scipy sparse .npz
    try:
        return load_npz(npz).tocsr().astype(np.float64)
    except Exception:
        pass
    # 2) numpy archive (np.savez / np.savez_compressed)
    try:
        z = np.load(npz, allow_pickle=True)
    except Exception as e:
        raise SystemExit(
            f"cannot open {npz} ({type(e).__name__}: {e}). If it lives on "
            f"Dropbox/iCloud, make sure it is downloaded locally (not an "
            f"online-only placeholder). Otherwise tell me how "
            f"build_cooccurrence.py writes it.")
    keys = set(getattr(z, "files", []))
    shape = tuple(int(x) for x in z["shape"]) if "shape" in keys else None
    if {"data", "indices", "indptr"} <= keys and shape:
        return csr_matrix((z["data"], z["indices"], z["indptr"]),
                          shape=shape).astype(np.float64)
    for d, r, c in [("data", "row", "col"), ("data", "rows", "cols"),
                    ("values", "row", "col"), ("counts", "row", "col")]:
        if {d, r, c} <= keys:
            return coo_matrix((z[d], (z[r], z[c])), shape=shape).tocsr().astype(np.float64)
    if len(keys) == 1:
        arr = np.asarray(z[next(iter(keys))], dtype=np.float64)
        if arr.ndim == 2:
            return csr_matrix(arr)
    raise SystemExit(f"unrecognised .npz layout in {npz}; keys={sorted(keys)}. "
                     f"Tell me how build_cooccurrence.py saves the matrix.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cooc-dir", help="dir with cooccurrence.npz and vocab.txt")
    ap.add_argument("--cooc", help="co-occurrence .npz (overrides --cooc-dir)")
    ap.add_argument("--vocab", help="vocab.txt (overrides --cooc-dir)")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--shift", type=float, default=1.0,
                    help="PMI shift: subtract log(shift); 1.0 = plain PPMI")
    ap.add_argument("--cds", type=float, default=0.75,
                    help="context distribution smoothing exponent; 1.0 = off")
    ap.add_argument("--sv-power", "--eig-weight", type=float, default=0.5,
                    dest="sv_power", help="U * Sigma**p (Levy: 0.5)")
    ap.add_argument("--combine", choices=["sum", "word", "concat"], default="sum",
                    help="sum = W+C (joint); word = W; concat = [W,C]")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--header", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cooc = a.cooc or (os.path.join(a.cooc_dir, "cooccurrence.npz") if a.cooc_dir else None)
    vocabp = a.vocab or (os.path.join(a.cooc_dir, "vocab.txt") if a.cooc_dir else None)
    if not cooc or not vocabp:
        raise SystemExit("give --cooc-dir, or both --cooc and --vocab")

    vocab = load_vocab(vocabp)
    X = load_cooc(cooc)
    V = X.shape[0]
    if len(vocab) != V:
        raise SystemExit(f"vocab {len(vocab)} != matrix dim {V}; check alignment")
    if a.combine == "concat" and a.dim % 2:
        raise SystemExit("--combine concat needs even --dim (dim/2 per factor)")
    print(f"ppmi-svd: V={V:,}, nnz={X.nnz:,}, dim={a.dim}, shift={a.shift}, "
          f"cds={a.cds}, sv_power={a.sv_power}, combine={a.combine}")

    # --- PMI with context-distribution smoothing --------------------------
    # PMI(i,j) = log( P(i,j) / (P(i) * P_cds(j)) ) - log(shift)
    # P_cds(j) ∝ count(j)**cds ; cds<1 damps frequent contexts (Levy et al.)
    row = np.asarray(X.sum(axis=1)).ravel()          # word marginal
    col = np.asarray(X.sum(axis=0)).ravel()          # context marginal
    total = row.sum()
    col_cds = col ** a.cds
    col_cds_sum = col_cds.sum()

    Xc = X.tocoo()
    with np.errstate(divide="ignore"):
        pmi = (np.log(Xc.data) + np.log(total * (col_cds_sum / total))  # = log(X_ij) + log(col_cds_sum)
               - np.log(row[Xc.row]) - np.log(col_cds[Xc.col]) - np.log(a.shift))
    # PPMI: keep positives only
    keep = pmi > 0
    from scipy.sparse import csr_matrix
    M = csr_matrix((pmi[keep], (Xc.row[keep], Xc.col[keep])), shape=(V, V))
    print(f"  PPMI nnz={M.nnz:,} ({100*M.nnz/max(X.nnz,1):.1f}% of cooc cells kept)")

    # --- truncated SVD ----------------------------------------------------
    k = a.dim if a.combine != "concat" else a.dim // 2
    np.random.seed(a.seed)
    try:
        from sklearn.utils.extmath import randomized_svd
        U, S, Vt = randomized_svd(M, n_components=k, n_iter=7,
                                  random_state=a.seed)
    except Exception:
        from scipy.sparse.linalg import svds
        U, S, Vt = svds(M.asfptype(), k=k)
        order = np.argsort(S)[::-1]
        U, S, Vt = U[:, order], S[order], Vt[order]

    Sp = S ** a.sv_power
    W = U * Sp                      # word factor  (V,k)
    C = (Vt.T) * Sp                 # context factor (V,k)
    if a.combine == "sum":
        E = W + C
    elif a.combine == "word":
        E = W
    else:
        E = np.concatenate([W, C], axis=1)
    print(f"  singular values: top={S[0]:.3f} ... tail={S[-1]:.3f}; emb {E.shape}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        if a.header:
            f.write(f"{V} {E.shape[1]}\n")
        for w, r in zip(vocab, E):
            f.write(w + " " + " ".join(f"{x:.6f}" for x in r) + "\n")
    print(f"wrote {V} x {E.shape[1]} (ppmi-svd/{a.combine}) -> {a.out}")


if __name__ == "__main__":
    main()
