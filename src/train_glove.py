#!/usr/bin/env python3
"""Train GloVe from a precomputed sparse co-occurrence matrix (PyTorch).

The model and loss are the standard log-bilinear GloVe. The only knobs that
matter for the dynamic-window study live upstream in build_cooccurrence.py; here
the two things to watch are x_max and f(x):

  * Budget-skipping removes the stopword-dominated cells that f(X) was invented
    to clip, so the top of the X distribution compresses. Stock x_max=100 will
    then clip too hard -- default is to DERIVE x_max from a high percentile of
    the actual nonzero X (usually lands lower). Override with --xmax.
  * --flat-f sets f==1, so you can test whether f(x) still matters once the
    window is doing the stopword work.

Emits GloVe text format (word v1 .. vd), readable by the standard eval suite.

--combine controls how the target (W) and context (W~) vector sets become the
final embedding:
  sum    -> W + W~   (standard GloVe; use with a SYMMETRIC matrix)
  concat -> [W | W~] (LATERALISED; use with an ASYMMETRIC X_R from
            build_cooccurrence.py --no-symmetrize, so W and W~ hold the right- and
            left-context roles instead of collapsing). concat doubles the output
            width, so for a dim-matched comparison run lateral at --dim d//2
            against symmetric-sum at --dim d.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")            # macOS ARM64
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GloVe(nn.Module):                        # the ~15 lines that matter
    def __init__(self, V, dim):
        super().__init__()
        self.w = nn.Embedding(V, dim)
        self.wc = nn.Embedding(V, dim)
        self.b = nn.Embedding(V, 1)
        self.bc = nn.Embedding(V, 1)
        for e in (self.w, self.wc):
            nn.init.uniform_(e.weight, -0.5 / dim, 0.5 / dim)
        for e in (self.b, self.bc):
            nn.init.zeros_(e.weight)

    def forward(self, i, j):
        dot = (self.w(i) * self.wc(j)).sum(1)
        return dot + self.b(i).squeeze(1) + self.bc(j).squeeze(1)


def load_vocab(path):
    words = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            words.append(line.rstrip("\n").split("\t")[0])
    return words


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cooc-dir", default="cooc")
    ap.add_argument("--out", default="glove_vectors.txt")
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.05)          # Adagrad, GloVe default
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--xmax", type=float, default=None, help="explicit; else derived")
    ap.add_argument("--xmax-percentile", type=float, default=99.0)
    ap.add_argument("--flat-f", action="store_true", help="f==1 ablation")
    ap.add_argument("--combine", choices=["sum", "concat"], default="sum",
                    help="sum=W+W~ (symmetric GloVe); concat=[W|W~] (lateralised, "
                         "use with --no-symmetrize matrix)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    cdir = Path(args.cooc_dir)

    X = sparse.load_npz(cdir / "cooccurrence.npz").tocoo()
    words = load_vocab(cdir / "vocab.txt")
    V = len(words)

    i = torch.as_tensor(X.row, dtype=torch.long)
    j = torch.as_tensor(X.col, dtype=torch.long)
    x = torch.as_tensor(X.data, dtype=torch.float32)
    logx = torch.log(x)

    xmax = args.xmax if args.xmax is not None else float(
        np.percentile(X.data, args.xmax_percentile))
    if args.flat_f:
        fx = torch.ones_like(x)
    else:
        fx = torch.clamp((x / xmax) ** args.alpha, max=1.0)
    print(f"V={V} nnz={len(x)} xmax={xmax:.2f} device={device} flat_f={args.flat_f}")

    i, j, x, logx, fx = (t.to(device) for t in (i, j, x, logx, fx))
    model = GloVe(V, args.dim).to(device)
    opt = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    n = len(x)

    for ep in range(args.epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for s in range(0, n, args.batch_size):
            b = perm[s:s + args.batch_size]
            pred = model(i[b], j[b])
            loss = (fx[b] * (pred - logx[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        print(f"epoch {ep + 1:3d}  loss {total / n:.5f}")

    W = model.w.weight.detach().cpu().numpy()
    Wc = model.wc.weight.detach().cpu().numpy()
    if args.combine == "concat":
        emb = np.concatenate([W, Wc], axis=1)          # [right-role | left-role]
    else:
        emb = W + Wc                                    # standard GloVe sum
    out_dim = emb.shape[1]
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as fh:
        for wi in range(V):
            fh.write(words[wi] + " " + " ".join(f"{v:.6f}" for v in emb[wi]) + "\n")
    print(f"wrote {V} x {out_dim} vectors (dim={args.dim}, combine={args.combine}) -> {outp}")


if __name__ == "__main__":
    main()
