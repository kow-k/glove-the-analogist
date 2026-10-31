#!/usr/bin/env python3
"""CBOW with NEGATIVE SAMPLING (PyTorch + Adam), warm/cold, seeded.

Sibling of the full-softmax CBOW. Same objective shape (predict the target from
the averaged context) but the O(V) softmax is replaced by SGNS-style negative
sampling: for each target, score the true target plus k noise words drawn from
the unigram^0.75 distribution. This is the trainer the §7 computational-cost
argument needs — cost per target drops from O(V) to O(k) — and the open question
it answers is whether NS-CBOW *converges to the same quality* as the full-softmax
CBOW (~0.5 STS). gensim's NS-CBOW did not, but that was SGD + alpha-decay with
~1/window the updates; here we use Adam and control epochs, as in the softmax
trainer.

Why Adam, not gensim's SGD: gensim floored CBOW at ~0.09 STS because its linear-
alpha SGD gives CBOW far fewer effective updates than SG. Adam's per-parameter
adaptive step is what let the full-softmax PyTorch CBOW reach ~0.5; we keep it
here so the ONLY change vs that trainer is softmax→NS.

Warm start injects the GloVe .vec into the INPUT (word) embeddings only, vocab-
aligned and rescaled to the table's init norm (identical to the softmax trainer);
--cold is the matched random-init baseline (byte-identical per --seed). Saves the
INPUT embeddings as the word vectors (same as the softmax trainer), headerless by
default. CLI mirrors the `--model cbow` branch of train_warmstart_word2vec.py so
it slots into the same batch script by swapping the tool name.

Examples (warm vs cold):
  python train_cbow_ns.py --corpus X.txt --glove glove-d128.vec --dim 128 \
      --window 10 --epochs 7 --negative 5 --min-count 5 --seed 1 --out cbowns-warm-r1.vec
  python train_cbow_ns.py --corpus X.txt --cold --dim 128 --window 10 \
      --epochs 7 --negative 5 --min-count 5 --seed 1 --out cbowns-cold-r1.vec
"""
import argparse, os, time, random, re
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k): return x


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def write_vec(path, vocab, emb, dim, header):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write(f"{len(vocab)} {dim}\n")
        for w, r in zip(vocab, emb):
            f.write(w + " " + " ".join(f"{x:.6f}" for x in r) + "\n")


def epoch_path(out, ep):
    """Per-epoch checkpoint name: replace the -ep<N> token with -ep<NN>, keeping
    a trailing -r<seed> in place, so
        foo-w10-lr0005-ep10-r1.vec -> foo-w10-lr0005-ep01-r1.vec ...
        foo-w10-lr0005-ep10.vec    -> foo-w10-lr0005-ep01.vec ..."""
    base, ext = os.path.splitext(out)
    m = re.search(r"(-r\d+)$", base)          # protect a trailing seed token
    tail = m.group(1) if m else ""
    stem = base[:len(base) - len(tail)] if tail else base
    stem = re.sub(r"-ep\d+$", "", stem)       # drop the original -ep<N>
    return f"{stem}-ep{ep:02d}{tail}{ext}"


def step_path(out, step):
    """Per-UPDATE checkpoint name: drop -ep<N>, append -st<step> (zero-padded),
    keep a trailing -r<seed>. The update axis is corpus/lr-invariant: the
    overshoot peaks at a fixed number of updates, not a fixed epoch."""
    base, ext = os.path.splitext(out)
    m = re.search(r"(-r\d+)$", base)
    tail = m.group(1) if m else ""
    stem = base[:len(base) - len(tail)] if tail else base
    stem = re.sub(r"-ep\d+$", "", stem)
    return f"{stem}-st{step:08d}{tail}{ext}"


def load_glove(path, dim):
    vecs = {}
    with open(path, encoding="utf-8") as fh:
        first = fh.readline().split()
        if not (len(first) == 2 and all(t.lstrip("-").isdigit() for t in first)):
            vecs[first[0]] = np.asarray(first[1:], dtype=np.float32)
        for line in fh:
            p = line.split()
            if len(p) >= 3:
                vecs[p[0]] = np.asarray(p[1:], dtype=np.float32)
    d = next(iter(vecs.values())).shape[0]
    if d != dim:
        raise SystemExit(f"GloVe dim {d} != --dim {dim}")
    return vecs


class CBOWDataset(Dataset):
    """(context_padded[2W], length, target); 0 = <UNK> = padding (excluded)."""
    def __init__(self, idx, window):
        W = window; n = len(idx); self.data = []
        for i in range(n):
            t = idx[i]
            if t == 0:
                continue
            lo, hi = max(0, i - W), min(n, i + W + 1)
            ctx = np.concatenate([idx[lo:i], idx[i + 1:hi]])
            if ctx.size == 0:
                continue
            padded = np.zeros(2 * W, dtype=np.int64)
            L = min(ctx.size, 2 * W); padded[:L] = ctx[:L]
            self.data.append((padded, t))
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        p, t = self.data[i]
        return torch.from_numpy(p), torch.tensor(t, dtype=torch.long)


class CBOWNS(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.inp = nn.Embedding(V, d)          # word (input) vectors — saved
        self.out = nn.Embedding(V, d)          # context/output (syn1neg analog)
        nn.init.uniform_(self.inp.weight, -0.5 / d, 0.5 / d)
        nn.init.zeros_(self.out.weight)        # word2vec inits output to 0

    def context_mean(self, ctx):
        e = self.inp(ctx)                      # (B,2W,d)
        m = (ctx != 0).float().unsqueeze(-1)   # (B,2W,1)
        return (e * m).sum(1) / m.sum(1).clamp(min=1.0)   # (B,d)

    def forward(self, ctx, target, negs):
        h = self.context_mean(ctx)                          # (B,d)
        v_pos = self.out(target)                            # (B,d)
        s_pos = (h * v_pos).sum(1)                           # (B,)
        v_neg = self.out(negs)                               # (B,k,d)
        s_neg = torch.bmm(v_neg, h.unsqueeze(2)).squeeze(2)  # (B,k)
        loss = -(F.logsigmoid(s_pos) + F.logsigmoid(-s_neg).sum(1))
        return loss.mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--glove", default=None, help="GloVe .vec to warm-start input embeddings")
    ap.add_argument("--cold", action="store_true", help="random init baseline")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--negative", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.010)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", choices=["match", "unit", "raw"], default="match")
    ap.add_argument("--header", action="store_true")
    ap.add_argument("--save-each-epoch", action="store_true",
                    help="also write a checkpoint .vec after every epoch "
                         "(<out stem with -ep<N> stripped>-ep01.vec ... -epNN.vec) "
                         "for an epoch-trajectory sweep; one training run, no retraining")
    ap.add_argument("--save-every-steps", type=int, default=0,
                    help="also checkpoint every N optimizer updates "
                         "(...-st00000500-r<seed>.vec ...). The update axis is "
                         "corpus/lr-invariant (the overshoot peaks at a fixed "
                         "number of updates, not a fixed epoch). 0 = off.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.cold and not a.glove:
        raise SystemExit("give --glove for warm start, or --cold for the baseline")

    set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if getattr(torch.backends, "mps", None)
                       and torch.backends.mps.is_available() else "cpu")

    # corpus + vocab
    sents = []
    with open(a.corpus, encoding="utf-8", errors="replace") as f:
        for line in f:
            tk = line.strip().lower().split()
            if tk: sents.append(tk)
    tokens = [t for s in sents for t in s]
    counter = Counter(tokens)
    vocab = ["<UNK>"] + [w for w, c in counter.items() if c >= a.min_count]
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    idx = np.array([w2i.get(t, 0) for t in tokens], dtype=np.int64)
    print(f"cbow-ns: {len(tokens):,} tokens, vocab {V:,}, device={dev}, "
          f"k={a.negative}, lr={a.lr}, {'cold' if a.cold else 'warm'}")

    # negative-sampling distribution: unigram^0.75, UNK excluded.
    # Precompute a word2vec-style unigram TABLE and sample by O(1) randint
    # indexing — avoids a per-batch torch.multinomial, which on MPS falls back
    # to CPU with a host sync every step and dominates wall-clock.
    freq = np.zeros(V, dtype=np.float64)
    for w, i in w2i.items():
        freq[i] = counter.get(w, 0)
    freq[0] = 0.0
    noise = freq ** 0.75
    noise = noise / noise.sum()
    TABLE = 1_000_000
    cum = np.cumsum(noise)
    table = np.clip(np.searchsorted(cum, (np.arange(TABLE) + 0.5) / TABLE), 0, V - 1)
    table_t = torch.as_tensor(table, dtype=torch.long, device=dev)

    model = CBOWNS(V, a.dim).to(dev)

    # warm-start input embeddings (same rescale as the softmax trainer)
    if not a.cold:
        glove = load_glove(a.glove, a.dim)
        with torch.no_grad():
            arr = model.inp.weight.detach().cpu().numpy()
        target = float(np.mean(np.linalg.norm(arr, axis=1))) or (0.5/np.sqrt(3))/np.sqrt(a.dim)
        G = np.stack(list(glove.values()))
        if a.scale == "raw":
            factor, gv = 1.0, glove
        elif a.scale == "unit":
            factor = target; gv = {w: v/(np.linalg.norm(v)+1e-8) for w, v in glove.items()}
        else:
            factor = target / (float(np.mean(np.linalg.norm(G, axis=1))) + 1e-12); gv = glove
        out = arr.copy(); hit = 0
        for w, i in w2i.items():
            v = gv.get(w)
            if v is not None:
                out[i] = v * factor; hit += 1
        with torch.no_grad():
            model.inp.weight.copy_(torch.from_numpy(out))
        print(f"  [warm] injected {hit}/{V} ({100*hit/V:.1f}%) factor≈{factor:.4g}")
    else:
        print("  [cold] random init")

    g = torch.Generator(); g.manual_seed(a.seed)
    loader = DataLoader(CBOWDataset(idx, a.window), batch_size=a.batch_size,
                        shuffle=True, num_workers=0, drop_last=True, generator=g)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    losses = []
    gstep = 0
    for ep in range(a.epochs):
        model.train(); tot = nb = 0; t0 = time.time()
        for ctx, tgt in tqdm(loader, desc=f"ep{ep+1}/{a.epochs}", leave=False,
                             ncols=72, bar_format="{desc} {percentage:3.0f}%|{bar}|"):
            ctx, tgt = ctx.to(dev), tgt.to(dev)
            negs = table_t[torch.randint(0, table_t.size(0),
                                         (ctx.size(0), a.negative), device=dev)]
            opt.zero_grad()
            loss = model(ctx, tgt, negs)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1; gstep += 1
            if a.save_every_steps and gstep % a.save_every_steps == 0:
                with torch.no_grad():
                    emb_s = model.inp.weight.cpu().numpy()
                write_vec(step_path(a.out, gstep), vocab, emb_s, a.dim, a.header)
        losses.append(tot/max(nb, 1))
        print(f"  [cbow-ns] ep{ep+1}/{a.epochs} loss={losses[-1]:.4f} time={time.time()-t0:.1f}s")
        if a.save_each_epoch:
            with torch.no_grad():
                emb_ep = model.inp.weight.cpu().numpy()
            cp = epoch_path(a.out, ep + 1)
            write_vec(cp, vocab, emb_ep, a.dim, a.header)
            print(f"    [checkpoint] ep{ep+1:02d} -> {cp}")

    with torch.no_grad():
        emb = model.inp.weight.cpu().numpy()
    write_vec(a.out, vocab, emb, a.dim, a.header)
    print(f"wrote {V} x {a.dim} ({'cold' if a.cold else 'warm'}, cbow-ns) -> {a.out}")
    print("  loss: " + " → ".join(f"{l:.4f}" for l in losses))


if __name__ == "__main__":
    main()
