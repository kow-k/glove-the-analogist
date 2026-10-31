#!/usr/bin/env python3
"""Plain CBOW trainer (bare minimum) with optional GloVe warm-start.

A deliberately stripped-down descendant of train_dynamic_lateral_cbow.py, built
for ONE job: measuring warm-start (GloVe-initialised) vs cold (random) under an
otherwise identical, confound-free CBOW. Everything that changes the context
geometry in the parent trainer — dynamic/peak windowing, lateralization,
cross-talk, L/R attention, three-way, IDF weighting, median aggregation — is
removed on purpose, because any of them would make warm-vs-cold uninterpretable.

What remains is the plain-baseline path of the parent: fixed symmetric window,
blind mean-pooled context, full-softmax cross-entropy, Adam. That is the path
that converges (~0.5 STS), so it is the honest testbed gensim CBOW could not be.

Added vs the parent:
  * --seed          deterministic init + shuffling (parent had none; required
                    for a byte-identical cold path and clean multi-seed runs)
  * --warm-init G   inject rescaled GloVe input vectors before training; absent
                    => cold baseline (byte-identical to the parent's plain init
                    for the same seed).

Metrics: NONE here. Save the .vec and evaluate it with
src/batch-evaluate-to-json.sh (src/evaluate_embeddings.py).

Examples (warm vs cold, matched):
  python src/train_warmstart_cbow.py corpus.txt --dim 64 --window 7 --epochs 10 \
      --seed 1 --warm-init glove-d064.vec --output cbow-warm-d064-w7-s1.vec
  python src/train_warmstart_cbow.py corpus.txt --dim 64 --window 7 --epochs 10 \
      --seed 1                       --output cbow-cold-d064-w7-s1.vec
"""
import argparse
import math
import os
import time
import logging
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── reproducibility ────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── corpus & vocab (word mode only) ────────────────────────────────
def load_corpus(path):
    sentences = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            toks = line.strip().lower().split()
            if toks:
                sentences.append(toks)
    tokens = [t for s in sentences for t in s]
    logger.info(f"  Tokens: {len(tokens):,}  Lines: {len(sentences):,}")
    return tokens


def build_vocab(tokens, min_count=5):
    counter = Counter(tokens)
    vocab = ["<UNK>"] + [w for w, c in counter.items() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(vocab)}
    logger.info(f"  Vocabulary: {len(vocab):,} (min_count={min_count})")
    return vocab, word2idx


# ── GloVe warm-start (inlined; keeps this file self-contained) ──────
def _load_glove(path, dim):
    """Read a .vec (word2vec text); auto-skip a `n dim` header. -> dict."""
    vecs = {}
    with open(path, encoding="utf-8") as fh:
        first = fh.readline().split()
        if not (len(first) == 2 and all(t.lstrip("-").isdigit() for t in first)):
            vecs[first[0]] = np.asarray(first[1:], dtype=np.float32)
        for line in fh:
            p = line.split()
            if len(p) >= 3:
                vecs[p[0]] = np.asarray(p[1:], dtype=np.float32)
    if not vecs:
        raise SystemExit(f"no vectors read from {path}")
    d = next(iter(vecs.values())).shape[0]
    if d != dim:
        raise SystemExit(f"GloVe dim {d} != --dim {dim}")
    return vecs


def warmstart_input_embeddings(weight, word2idx, glove_path, scale="match"):
    """Overwrite in-vocab rows of `weight` (nn.Parameter) with rescaled GloVe.

    Rescales the whole GloVe matrix by one factor so its mean row-norm equals
    the table's current init mean row-norm (preserves GloVe geometry, fixes
    scale). Words absent from GloVe keep their random init. Returns info dict.
    """
    with torch.no_grad():
        arr = weight.detach().cpu().numpy()
    V, dim = arr.shape
    glove = _load_glove(glove_path, dim)
    G = np.stack(list(glove.values()))

    target_norm = float(np.mean(np.linalg.norm(arr, axis=1)))
    if not np.isfinite(target_norm) or target_norm == 0.0:
        target_norm = (0.5 / np.sqrt(3)) / np.sqrt(dim)

    if scale == "raw":
        factor = 1.0
        glove_s = glove
    elif scale == "unit":
        factor = target_norm
        glove_s = {w: v / (np.linalg.norm(v) + 1e-8) * factor
                   for w, v in glove.items()}
    else:  # match
        glove_norm = float(np.mean(np.linalg.norm(G, axis=1)))
        factor = target_norm / (glove_norm + 1e-12)
        glove_s = {w: v * factor for w, v in glove.items()}

    out = arr.copy()
    n_hit = 0
    for w, i in word2idx.items():
        v = glove_s.get(w)
        if v is not None and 0 <= i < V:
            out[i] = v
            n_hit += 1
    if not np.all(np.isfinite(out)):
        raise SystemExit("warm-start produced non-finite rows; check GloVe/scale")
    with torch.no_grad():
        weight.copy_(torch.from_numpy(out))
    return {"n_injected": n_hit, "vocab": len(word2idx),
            "coverage": n_hit / max(len(word2idx), 1),
            "factor": factor, "target_norm": target_norm}


# ── dataset: plain fixed-window CBOW, blind context ─────────────────
class CBOWDataset(Dataset):
    """(context_padded[2W], length, target). 0 = <UNK> = padding (excluded)."""

    def __init__(self, tokens, word2idx, window):
        self.window = window
        idx = np.array([word2idx.get(t, 0) for t in tokens], dtype=np.int64)
        n = len(idx)
        W = window
        self.data = []
        for i in range(n):
            target = idx[i]
            if target == 0:                      # skip UNK targets
                continue
            lo, hi = max(0, i - W), min(n, i + W + 1)
            ctx = np.concatenate([idx[lo:i], idx[i + 1:hi]])
            if ctx.size == 0:
                continue
            padded = np.zeros(2 * W, dtype=np.int64)
            L = min(ctx.size, 2 * W)
            padded[:L] = ctx[:L]
            self.data.append((padded, L, target))
        logger.info(f"  Dataset: {len(self.data):,} samples (window=±{window})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        padded, L, target = self.data[i]
        return (torch.from_numpy(padded),
                torch.tensor(L, dtype=torch.long),
                torch.tensor(target, dtype=torch.long))


# ── model: plain CBOW, mean-pool, full-softmax ──────────────────────
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        nn.init.uniform_(self.embeddings.weight, -0.5 / dim, 0.5 / dim)
        nn.init.uniform_(self.output.weight, -0.5 / dim, 0.5 / dim)

    def forward(self, ctx, lengths):
        emb = self.embeddings(ctx)                       # (B, 2W, D)
        mask = (ctx != 0).float().unsqueeze(-1)          # (B, 2W, 1)
        summed = (emb * mask).sum(dim=1)                 # (B, D)
        cnt = mask.sum(dim=1).clamp(min=1.0)             # (B, 1)
        avg = summed / cnt
        return self.output(avg)


def train(model, dataset, args, device, label):
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True, generator=g)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    losses = []
    for epoch in range(args.epochs):
        model.train()
        if args.lr_decay == "linear":
            lr = args.lr * (1.0 - epoch / args.epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        else:
            lr = args.lr
        total, nb, t0 = 0.0, 0, time.time()
        pbar = tqdm(loader, desc=f"ep{epoch+1}/{args.epochs}", leave=False,
                    ncols=72, bar_format="{desc} {percentage:3.0f}%|{bar}|")
        for ctx, lengths, target in pbar:
            ctx, lengths, target = ctx.to(device), lengths.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ctx, lengths), target)
            loss.backward()
            optimizer.step()
            total += loss.item(); nb += 1
        avg = total / max(nb, 1)
        losses.append(avg)
        logger.info(f"  [{label}] ep{epoch+1}/{args.epochs} loss={avg:.4f} "
                    f"lr={lr:.5f} time={time.time()-t0:.1f}s")
    model.cpu()
    return losses


def save_embeddings(emb, vocab, path, header=True):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    V, D = emb.shape
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write(f"{V} {D}\n")
        for i in range(V):
            f.write(vocab[i] + " " + " ".join(f"{x:.6f}" for x in emb[i]) + "\n")
    logger.info(f"  Saved: {path} ({V} words, {D}d, header={header})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--window", type=int, default=7, help="fixed ±N context window")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.010)
    ap.add_argument("--lr-decay", choices=["none", "linear"], default="none")
    ap.add_argument("--seed", type=int, default=1)
    # warm start
    ap.add_argument("--warm-init", default=None,
                    help="GloVe .vec to initialise input embeddings; absent = cold")
    ap.add_argument("--scale", choices=["match", "unit", "raw"], default="match")
    # output
    ap.add_argument("--output", required=True, help="embeddings out (.vec)")
    ap.add_argument("--no-header", action="store_true",
                    help="write headerless .vec (default writes `V D` header)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if getattr(torch.backends, "mps", None)
                          and torch.backends.mps.is_available() else "cpu")
    mode = "cold" if not args.warm_init else "warm"
    logger.info(f"Device: {device} | mode={mode} | dim={args.dim} window=±{args.window} "
                f"epochs={args.epochs} lr={args.lr} decay={args.lr_decay} seed={args.seed}")

    tokens = load_corpus(args.corpus)
    vocab, word2idx = build_vocab(tokens, min_count=args.min_count)
    dataset = CBOWDataset(tokens, word2idx, window=args.window)

    model = CBOWModel(len(vocab), args.dim)      # random init (seeded)
    if args.warm_init:
        info = warmstart_input_embeddings(model.embeddings.weight, word2idx,
                                          args.warm_init, scale=args.scale)
        logger.info(f"  [warm] injected {info['n_injected']}/{info['vocab']} "
                    f"({100*info['coverage']:.1f}%) factor={info['factor']:.4g} "
                    f"target_norm={info['target_norm']:.4g} scale={args.scale}")
    else:
        logger.info("  [cold] random init (no warm-start)")

    label = f"cbow-{mode}-d{args.dim:03d}-w{args.window}-ep{args.epochs}-s{args.seed}"
    losses = train(model, dataset, args, device, label)

    with torch.no_grad():
        emb = model.embeddings.weight.cpu().numpy()
    save_embeddings(emb, vocab, args.output, header=not args.no_header)
    logger.info(f"  Loss: {' → '.join(f'{l:.4f}' for l in losses)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
