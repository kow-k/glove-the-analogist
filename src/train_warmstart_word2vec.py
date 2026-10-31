#!/usr/bin/env python3
"""Warm-start CBOW / Skip-gram from GloVe vectors, with a matched cold baseline.

The clean measurement is warm-vs-cold *within the same trainer*: run --cold and
warm (default) with identical hyperparameters/seed, and the difference is the
warm-start effect.

Three engines, chosen by --model, sharing one CLI (so CBOW-softmax, CBOW-NS and
Skip-gram can be traced under one pipeline):
  * --model sg     -> gensim Skip-gram (SGNS). Converges fine on this data; the
                      warm-vs-cold washout result for SG came from here.
  * --model cbow   -> custom full-softmax PyTorch CBOW (Adam). gensim's CBOW is
                      undertrained on small corpora (STS/SL-999 floored), so it
                      is routed to this converging trainer. Plain baseline: fixed
                      symmetric window, blind mean-pooled context, cross-entropy
                      — no dynamic windowing, lateralization, IDF, attention, or
                      median (those would confound warm-vs-cold).
  * --model cbowns -> same averaged-context CBOW but with negative sampling
                      instead of the O(V) softmax (Adam; --negative noise words
                      from the unigram^0.75 table). The cheap engine that reaches
                      full-softmax quality; shares the cbow CLI.

Warm start (both engines): the input (word) vectors are initialised from a
GloVe .vec, rescaled to the trainer's own init scale so the injected geometry
doesn't blow up, and aligned by vocabulary (words absent from GloVe keep the
random init). Words present keep GloVe's *relative* geometry; only the overall
scale is fixed. --cold skips injection (byte-identical to the random init for a
given --seed, for the cbow engine).

Output: headerless `word v1..vd` by default (use --header for the `n dim` line);
identical format from both engines, so one eval/aggregate pipeline consumes both.

Engine-specific flags:
  * gensim-only : --negative, --sample, --glove-context (ignored, with a note,
                  under --model cbow).
  * cbow-only   : --batch-size, --lr-decay (ignored by the gensim engine).
  * --lr        : per-engine default when omitted — 0.025 (sg, gensim) /
                  0.010 (cbow, Adam). For cbow this is the Adam LR (~0.01),
                  NOT gensim's 0.025 — LRs do not transfer across engines.

Examples (warm vs cold):
  # Skip-gram (gensim)
  python warmstart_word2vec.py --corpus X.txt --model sg  --glove glove-d128.vec \
      --dim 128 --window 10 --epochs 5  --seed 1 --out sg-warm-r1.vec
  # CBOW (custom torch), warm then cold
  python warmstart_word2vec.py --corpus X.txt --model cbow --glove glove-d064.vec \
      --dim 64 --window 7 --epochs 10 --seed 1 --out cbow-warm-r1.vec
  python warmstart_word2vec.py --corpus X.txt --model cbow \
      --dim 64 --window 7 --epochs 10 --seed 1 --cold --out cbow-cold-r1.vec
"""
import argparse
import re
import numpy as np
from pathlib import Path


def epoch_path(out, ep):
    """Per-epoch checkpoint name: replace the -ep<N> token with -ep<NN>, keeping
    a trailing -r<seed> in place (foo-ep10-r1.vec -> foo-ep01-r1.vec ...)."""
    import os
    base, ext = os.path.splitext(out)
    m = re.search(r"(-r\d+)$", base)
    tail = m.group(1) if m else ""
    stem = base[:len(base) - len(tail)] if tail else base
    stem = re.sub(r"-ep\d+$", "", stem)
    return f"{stem}-ep{ep:02d}{tail}{ext}"


def step_path(out, step):
    """Per-UPDATE checkpoint name: drop the -ep<N> token and append -st<step>
    (zero-padded), keeping a trailing -r<seed> in place. The update axis is the
    corpus/lr-invariant one: the overshoot peaks at a fixed number of updates,
    whereas its epoch moves with tokens-per-epoch and learning rate."""
    import os
    base, ext = os.path.splitext(out)
    m = re.search(r"(-r\d+)$", base)
    tail = m.group(1) if m else ""
    stem = base[:len(base) - len(tail)] if tail else base
    stem = re.sub(r"-ep\d+$", "", stem)
    return f"{stem}-st{step:08d}{tail}{ext}"


# ─────────────────────────────────────────── shared: GloVe .vec loading + IO
def load_vectors(path):
    """Read a .vec (word2vec text). Auto-detects and skips a `n dim` header."""
    vecs = {}
    with open(path, encoding="utf-8") as fh:
        first = fh.readline().split()
        if len(first) == 2 and all(t.lstrip("-").isdigit() for t in first):
            pass                                    # header line, skip
        else:
            w, *xs = first
            vecs[w] = np.asarray(xs, dtype=np.float32)
        for line in fh:
            p = line.split()
            if len(p) < 3:
                continue
            vecs[p[0]] = np.asarray(p[1:], dtype=np.float32)
    return vecs


def write_vectors(words, matrix, path, header):
    """Write aligned (words, matrix[V,d]) as `word v1..vd`, optional header."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    V, dim = matrix.shape
    with open(path, "w", encoding="utf-8") as fh:
        if header:
            fh.write(f"{V} {dim}\n")
        for w, row in zip(words, matrix):
            fh.write(w + " " + " ".join(f"{x:.6f}" for x in row) + "\n")


def rescale_factor(glove_matrix, dim, target_norm, scale_mode):
    """One scalar so injected rows land at `target_norm` (mode 'match'/'unit')."""
    if scale_mode == "raw":
        return 1.0
    if scale_mode == "unit":
        return target_norm  # applied to L2-normalised rows
    cur = float(np.mean(np.linalg.norm(glove_matrix, axis=1)))  # 'match'
    return target_norm / (cur + 1e-12)


# ═══════════════════════════════════════════════════ ENGINE 1: gensim (sg)
def run_gensim(args):
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence

    lr = 0.025 if args.lr is None else args.lr
    model = Word2Vec(sg=1 if args.model == "sg" else 0, vector_size=args.dim,
                     window=args.window, min_count=args.min_count,
                     negative=args.negative, sample=args.sample, alpha=lr,
                     seed=args.seed, workers=1, epochs=args.epochs)
    model.build_vocab(corpus_iterable=LineSentence(args.corpus))

    if not args.cold:
        glove = load_vectors(args.glove)
        G = np.stack(list(glove.values()))
        target = (0.5 / np.sqrt(3)) / np.sqrt(args.dim)   # gensim mean row-norm
        factor = rescale_factor(G, args.dim, target, args.scale)
        gv = ({w: v / (np.linalg.norm(v) + 1e-8) for w, v in glove.items()}
              if args.scale == "unit" else glove)
        n_hit = 0
        for i, w in enumerate(model.wv.index_to_key):
            v = gv.get(w)
            if v is not None:
                if v.shape[0] != args.dim:
                    raise SystemExit(f"GloVe dim {v.shape[0]} != --dim {args.dim}")
                model.wv.vectors[i] = v * factor
                n_hit += 1
        print(f"warm-start (gensim {args.model}): injected {n_hit}/{len(model.wv)} "
              f"(scale={args.scale}, factor≈{factor:.4g})")
        if args.glove_context:
            gc = load_vectors(args.glove_context)
            Gc = np.stack(list(gc.values()))
            fc = rescale_factor(Gc, args.dim, target, args.scale)
            for i, w in enumerate(model.wv.index_to_key):
                v = gc.get(w)
                if v is not None:
                    model.syn1neg[i] = v * fc

    cbs = []
    if getattr(args, "save_each_epoch", False):
        from gensim.models.callbacks import CallbackAny2Vec

        class EpochSaver(CallbackAny2Vec):
            """Save the word vectors after every epoch for a trajectory sweep."""
            def __init__(self, out, header):
                self.out, self.header, self.ep = out, header, 0
            def on_epoch_end(self, model):
                self.ep += 1
                p = epoch_path(self.out, self.ep)
                write_vectors(list(model.wv.index_to_key),
                              np.asarray(model.wv.vectors), p, self.header)
                print(f"    [checkpoint] ep{self.ep:02d} -> {p}")

        cbs = [EpochSaver(args.out, args.header)]

    model.train(corpus_iterable=LineSentence(args.corpus),
                total_examples=model.corpus_count, epochs=model.epochs,
                callbacks=cbs)
    write_vectors(list(model.wv.index_to_key), np.asarray(model.wv.vectors),
                  args.out, args.header)
    print(f"wrote {len(model.wv)} x {args.dim} vectors "
          f"({'cold' if args.cold else 'warm'}, gensim {args.model}) -> {args.out}")


# ═══════════════════════════════════════════ ENGINE 2: custom CBOW (torch)
def run_cbow(args):
    import time
    import random
    from collections import Counter
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    try:
        from tqdm import tqdm
    except ImportError:                       # tqdm optional
        def tqdm(x, **k):
            return x

    if args.glove_context:
        print("[note] --glove-context is gensim-only; ignored under --model cbow")

    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    lr = 0.010 if args.lr is None else args.lr
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if getattr(torch.backends, "mps", None)
                          and torch.backends.mps.is_available() else "cpu")

    # corpus + vocab (word mode, <UNK>=0=padding)
    sentences = []
    with open(args.corpus, encoding="utf-8", errors="replace") as f:
        for line in f:
            toks = line.strip().lower().split()
            if toks:
                sentences.append(toks)
    tokens = [t for s in sentences for t in s]
    counter = Counter(tokens)
    vocab = ["<UNK>"] + [w for w, c in counter.items() if c >= args.min_count]
    word2idx = {w: i for i, w in enumerate(vocab)}
    print(f"cbow: {len(tokens):,} tokens, vocab {len(vocab):,} "
          f"(min_count={args.min_count}), device={device}, lr={lr}")

    # dataset: fixed +/-window, blind context
    idx = np.array([word2idx.get(t, 0) for t in tokens], dtype=np.int64)
    n, W = len(idx), args.window
    samples = []
    for i in range(n):
        t = idx[i]
        if t == 0:
            continue
        lo, hi = max(0, i - W), min(n, i + W + 1)
        ctx = np.concatenate([idx[lo:i], idx[i + 1:hi]])
        if ctx.size == 0:
            continue
        padded = np.zeros(2 * W, dtype=np.int64)
        L = min(ctx.size, 2 * W)
        padded[:L] = ctx[:L]
        samples.append((padded, t))
    print(f"cbow: {len(samples):,} samples (window=±{W})")

    class DS(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            p, t = samples[i]
            return torch.from_numpy(p), torch.tensor(t, dtype=torch.long)

    class Model(nn.Module):
        def __init__(self, V, d):
            super().__init__()
            self.embeddings = nn.Embedding(V, d)
            self.output = nn.Linear(d, V, bias=False)
            nn.init.uniform_(self.embeddings.weight, -0.5 / d, 0.5 / d)
            nn.init.uniform_(self.output.weight, -0.5 / d, 0.5 / d)

        def forward(self, ctx):
            emb = self.embeddings(ctx)
            mask = (ctx != 0).float().unsqueeze(-1)
            avg = (emb * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            return self.output(avg)

    model = Model(len(vocab), args.dim)

    # warm-start input embeddings
    if not args.cold:
        glove = load_vectors(args.glove)
        d0 = next(iter(glove.values())).shape[0]
        if d0 != args.dim:
            raise SystemExit(f"GloVe dim {d0} != --dim {args.dim}")
        with torch.no_grad():
            arr = model.embeddings.weight.detach().cpu().numpy()
        target = float(np.mean(np.linalg.norm(arr, axis=1)))
        if not np.isfinite(target) or target == 0.0:
            target = (0.5 / np.sqrt(3)) / np.sqrt(args.dim)
        G = np.stack(list(glove.values()))
        factor = rescale_factor(G, args.dim, target, args.scale)
        gv = ({w: v / (np.linalg.norm(v) + 1e-8) for w, v in glove.items()}
              if args.scale == "unit" else glove)
        out = arr.copy(); n_hit = 0
        for w, i in word2idx.items():
            v = gv.get(w)
            if v is not None:
                out[i] = v * factor; n_hit += 1
        if not np.all(np.isfinite(out)):
            raise SystemExit("warm-start produced non-finite rows; check GloVe/scale")
        with torch.no_grad():
            model.embeddings.weight.copy_(torch.from_numpy(out))
        print(f"warm-start (cbow): injected {n_hit}/{len(vocab)} "
              f"({100*n_hit/len(vocab):.1f}%, scale={args.scale}, "
              f"factor≈{factor:.4g}, target_norm≈{target:.4g})")
    else:
        print("cold: random init (no warm-start)")

    # train
    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(DS(), batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True, generator=g)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.to(device)
    losses = []
    gstep = 0
    for ep in range(args.epochs):
        model.train()
        if args.lr_decay == "linear":
            cur = lr * (1.0 - ep / args.epochs)
            for pg in opt.param_groups:
                pg["lr"] = cur
        else:
            cur = lr
        tot = nb = 0; t0 = time.time()
        for ctx, tgt in tqdm(loader, desc=f"ep{ep+1}/{args.epochs}", leave=False,
                             ncols=72, bar_format="{desc} {percentage:3.0f}%|{bar}|"):
            ctx, tgt = ctx.to(device), tgt.to(device)
            opt.zero_grad()
            loss = crit(model(ctx), tgt)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1; gstep += 1
            if args.save_every_steps and gstep % args.save_every_steps == 0:
                with torch.no_grad():
                    emb_s = model.embeddings.weight.cpu().numpy()
                write_vectors(vocab, emb_s, step_path(args.out, gstep), args.header)
        losses.append(tot / max(nb, 1))
        print(f"  [cbow] ep{ep+1}/{args.epochs} loss={losses[-1]:.4f} "
              f"lr={cur:.5f} time={time.time()-t0:.1f}s")
        if args.save_each_epoch:
            with torch.no_grad():
                emb_ep = model.embeddings.weight.cpu().numpy()
            p = epoch_path(args.out, ep + 1)
            write_vectors(vocab, emb_ep, p, args.header)
            print(f"    [checkpoint] ep{ep+1:02d} -> {p}")

    model.cpu()
    with torch.no_grad():
        emb = model.embeddings.weight.cpu().numpy()
    write_vectors(vocab, emb, args.out, args.header)
    print(f"wrote {len(vocab)} x {args.dim} vectors "
          f"({'cold' if args.cold else 'warm'}, cbow) -> {args.out}")
    print("  loss: " + " → ".join(f"{l:.4f}" for l in losses))


# ═══════════════════════════════════ ENGINE 3: custom CBOW + neg. sampling
def run_cbow_ns(args):
    """CBOW with negative sampling (SGNS-style objective), PyTorch + Adam.

    Same averaged-context input as the full-softmax CBOW, but the O(V) softmax
    is replaced by scoring the true target plus `--negative` noise words drawn
    from the unigram^0.75 table. Saves the INPUT (word) embeddings, identical
    format to the other engines; supports warm/cold and --save-each-epoch.
    """
    import time
    import random
    from collections import Counter
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    if args.glove_context:
        print("[note] --glove-context is gensim-only; ignored under --model cbowns")

    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    lr = 0.010 if args.lr is None else args.lr   # Adam; matches the softmax CBOW
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if getattr(torch.backends, "mps", None)
                          and torch.backends.mps.is_available() else "cpu")

    # corpus + vocab (<UNK>=0=padding, excluded as target and from noise)
    sentences = []
    with open(args.corpus, encoding="utf-8", errors="replace") as f:
        for line in f:
            toks = line.strip().lower().split()
            if toks:
                sentences.append(toks)
    tokens = [t for s in sentences for t in s]
    counter = Counter(tokens)
    vocab = ["<UNK>"] + [w for w, c in counter.items() if c >= args.min_count]
    word2idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    idx = np.array([word2idx.get(t, 0) for t in tokens], dtype=np.int64)
    print(f"cbow-ns: {len(tokens):,} tokens, vocab {V:,} "
          f"(min_count={args.min_count}), device={device}, "
          f"k={args.negative}, lr={lr}, {'cold' if args.cold else 'warm'}")

    # negative-sampling unigram table (unigram^0.75, UNK excluded); sampled by
    # O(1) randint indexing to avoid a per-batch torch.multinomial (which falls
    # back to CPU on MPS and dominates wall-clock).
    freq = np.zeros(V, dtype=np.float64)
    for w, i in word2idx.items():
        freq[i] = counter.get(w, 0)
    freq[0] = 0.0
    noise = freq ** 0.75
    noise = noise / noise.sum()
    TABLE = 1_000_000
    cum = np.cumsum(noise)
    table = np.clip(np.searchsorted(cum, (np.arange(TABLE) + 0.5) / TABLE), 0, V - 1)
    table_t = torch.as_tensor(table, dtype=torch.long, device=device)

    # dataset: fixed +/-window, blind context
    n, W = len(idx), args.window
    samples = []
    for i in range(n):
        t = idx[i]
        if t == 0:
            continue
        lo, hi = max(0, i - W), min(n, i + W + 1)
        ctx = np.concatenate([idx[lo:i], idx[i + 1:hi]])
        if ctx.size == 0:
            continue
        padded = np.zeros(2 * W, dtype=np.int64)
        L = min(ctx.size, 2 * W)
        padded[:L] = ctx[:L]
        samples.append((padded, t))
    print(f"cbow-ns: {len(samples):,} samples (window=±{W})")

    class DS(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            p, t = samples[i]
            return torch.from_numpy(p), torch.tensor(t, dtype=torch.long)

    class CBOWNS(nn.Module):
        def __init__(self, V, d):
            super().__init__()
            self.inp = nn.Embedding(V, d)      # word (input) vectors — saved
            self.out = nn.Embedding(V, d)      # context/output (syn1neg analog)
            nn.init.uniform_(self.inp.weight, -0.5 / d, 0.5 / d)
            nn.init.zeros_(self.out.weight)

        def context_mean(self, ctx):
            e = self.inp(ctx)
            m = (ctx != 0).float().unsqueeze(-1)
            return (e * m).sum(1) / m.sum(1).clamp(min=1.0)

        def forward(self, ctx, target, negs):
            h = self.context_mean(ctx)
            s_pos = (h * self.out(target)).sum(1)
            s_neg = torch.bmm(self.out(negs), h.unsqueeze(2)).squeeze(2)
            loss = -(F.logsigmoid(s_pos) + F.logsigmoid(-s_neg).sum(1))
            return loss.mean()

    model = CBOWNS(V, args.dim)

    # warm-start INPUT embeddings (same rescale as the other engines)
    if not args.cold:
        glove = load_vectors(args.glove)
        d0 = next(iter(glove.values())).shape[0]
        if d0 != args.dim:
            raise SystemExit(f"GloVe dim {d0} != --dim {args.dim}")
        with torch.no_grad():
            arr = model.inp.weight.detach().cpu().numpy()
        target = float(np.mean(np.linalg.norm(arr, axis=1)))
        if not np.isfinite(target) or target == 0.0:
            target = (0.5 / np.sqrt(3)) / np.sqrt(args.dim)
        G = np.stack(list(glove.values()))
        factor = rescale_factor(G, args.dim, target, args.scale)
        gv = ({w: v / (np.linalg.norm(v) + 1e-8) for w, v in glove.items()}
              if args.scale == "unit" else glove)
        out = arr.copy(); n_hit = 0
        for w, i in word2idx.items():
            v = gv.get(w)
            if v is not None:
                out[i] = v * factor; n_hit += 1
        if not np.all(np.isfinite(out)):
            raise SystemExit("warm-start produced non-finite rows; check GloVe/scale")
        with torch.no_grad():
            model.inp.weight.copy_(torch.from_numpy(out))
        print(f"warm-start (cbow-ns): injected {n_hit}/{V} "
              f"({100*n_hit/V:.1f}%, scale={args.scale}, factor≈{factor:.4g})")
    else:
        print("cold: random init (no warm-start)")

    model.to(device)
    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(DS(), batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True, generator=g)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    gstep = 0
    for ep in range(args.epochs):
        model.train(); tot = nb = 0; t0 = time.time()
        for ctx, tgt in tqdm(loader, desc=f"ep{ep+1}/{args.epochs}", leave=False,
                             ncols=72, bar_format="{desc} {percentage:3.0f}%|{bar}|"):
            ctx, tgt = ctx.to(device), tgt.to(device)
            negs = table_t[torch.randint(0, table_t.size(0),
                                         (ctx.size(0), args.negative), device=device)]
            opt.zero_grad()
            loss = model(ctx, tgt, negs)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1; gstep += 1
            if args.save_every_steps and gstep % args.save_every_steps == 0:
                with torch.no_grad():
                    emb_s = model.inp.weight.cpu().numpy()
                p = step_path(args.out, gstep)
                write_vectors(vocab, emb_s, p, args.header)
        losses.append(tot / max(nb, 1))
        print(f"  [cbow-ns] ep{ep+1}/{args.epochs} loss={losses[-1]:.4f} "
              f"time={time.time()-t0:.1f}s")
        if args.save_each_epoch:
            with torch.no_grad():
                emb_ep = model.inp.weight.cpu().numpy()
            p = epoch_path(args.out, ep + 1)
            write_vectors(vocab, emb_ep, p, args.header)
            print(f"    [checkpoint] ep{ep+1:02d} -> {p}")

    with torch.no_grad():
        emb = model.inp.weight.cpu().numpy()
    write_vectors(vocab, emb, args.out, args.header)
    print(f"wrote {V} x {args.dim} vectors "
          f"({'cold' if args.cold else 'warm'}, cbow-ns) -> {args.out}")
    print("  loss: " + " → ".join(f"{l:.4f}" for l in losses))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", choices=["cbow", "cbowns", "sg"], required=True,
                    help="cbow=full-softmax CBOW (torch); cbowns=negative-sampling "
                         "CBOW (torch); sg=Skip-gram (gensim SGNS)")
    ap.add_argument("--glove", default=None, help="GloVe .vec to warm-start from")
    ap.add_argument("--glove-context", default=None,
                    help="[gensim only] GloVe context .vec to warm output (syn1neg)")
    ap.add_argument("--cold", action="store_true", help="no warm start (baseline)")
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--window", type=int, default=10,
                    help="context window (±N); fixed for cbow")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--negative", type=int, default=5,
                    help="[sg + cbowns] number of negative samples")
    ap.add_argument("--sample", type=float, default=1e-3, help="[gensim only]")
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate; default per engine "
                         "(sg 0.025 / cbow 0.010 / cbowns 0.010)")
    ap.add_argument("--batch-size", type=int, default=512,
                    help="[cbow / cbowns only]")
    ap.add_argument("--lr-decay", choices=["none", "linear"], default="none",
                    help="[cbow only] Adam LR schedule")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", choices=["match", "unit", "raw"], default="match")
    ap.add_argument("--header", action="store_true", help="write word2vec header line")
    ap.add_argument("--save-each-epoch", action="store_true",
                    help="all engines: also write a checkpoint .vec after every "
                         "epoch (...-ep01-r<seed>.vec ...) for a trajectory sweep")
    ap.add_argument("--save-every-steps", type=int, default=0,
                    help="[cbow / cbowns] also checkpoint every N optimizer updates "
                         "(...-st00000500-r<seed>.vec ...). The update axis is "
                         "corpus/lr-invariant; use it to compare the overshoot "
                         "across corpus sizes and learning rates. 0 = off.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not args.cold and not args.glove:
        raise SystemExit("give --glove for warm start, or --cold for the baseline")

    if args.model == "cbow":
        run_cbow(args)
    elif args.model == "cbowns":
        run_cbow_ns(args)
    else:
        run_gensim(args)


if __name__ == "__main__":
    main()
