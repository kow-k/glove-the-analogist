#!/usr/bin/env python3
"""Fair wall-clock comparison: Skip-gram-only vs GloVe→CBOW sequence.

Measures the time to PRODUCE each deliverable at the configs actually used in
the experiments, so the "is the sequence cheap?" claim rests on a measured
number rather than intuition. Three timed pipelines:

  1. SG-only          : train_warmstart_word2vec.py --model sg --cold
  2. GloVe→CBOW seq   : build_cooccurrence.py  →  train_glove.py  →
                        train_warmstart_word2vec.py --model cbow --warm-init
  3. CBOW-cold-only   : train_warmstart_word2vec.py --model cbow --cold
                        (reference: if you only want STS/SICK you can skip the
                        GloVe stage entirely — this is the honest cheaper
                        alternative the sequence must be judged against)

Each pipeline is run --repeats times; medians are reported (timing is noisy).
The sequence total is timed per-repeat as build+train+cbow, then medianed.

FAIRNESS — read before trusting the numbers:
  * Device must match. Our torch trainers (GloVe, CBOW) auto-pick CUDA/MPS,
    while gensim SG is CPU-only. Comparing MPS-CBOW against CPU-SG is
    apples-to-oranges. For a clean comparison, force everything onto CPU
    (this script sets CUDA_VISIBLE_DEVICES="" and PYTORCH_MPS_DISABLE=1 in the
    child env by default; use --allow-accel to let the trainers use the GPU,
    and then state the device per stage in the paper).
  * Threads must match. gensim here runs workers=1 (single-thread). This script
    pins the torch/BLAS stages to --threads (default 1) via OMP/MKL env vars so
    SG and CBOW are compared single-thread-to-single-thread. Raise --threads
    only if you also raise gensim workers to match, and report both.
  * Epochs are as-used, NOT equalized — the comparison is "time to the model
    you actually ship," with its own epoch budget (SG ep5 / GloVe ep10-20 /
    CBOW ep7). That's the honest efficiency question; it is not a per-epoch
    speed benchmark.
  * The sequence cost INCLUDES the GloVe stage (build + train). That is the
    whole point — don't quote CBOW time alone as the sequence cost.

Example:
  python time_compare.py --corpus clean4a_30k.txt --dim 128 \
      --glove-window 30 --glove-epochs 20 --sg-window 15 --sg-epochs 5 \
      --cbow-window 10 --cbow-epochs 7 --repeats 3
"""
import argparse, os, sys, time, statistics, platform, subprocess, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_timed(cmd, env, log):
    log.write(f"\n$ {' '.join(map(str,cmd))}\n"); log.flush()
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        log.flush()
        try:
            tail = Path(log.name).read_text(errors="replace").splitlines()[-15:]
        except Exception:
            tail = []
        sys.stderr.write("\n--- stage failed ---\n$ " + " ".join(map(str, cmd))
                         + "\n" + "\n".join(tail) + f"\n(full log: {log.name})\n")
        raise SystemExit(1)
    return dt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1,
                    help="CPU threads for torch/BLAS stages (match gensim workers)")
    ap.add_argument("--allow-accel", action="store_true",
                    help="let torch trainers use CUDA/MPS (default: force CPU)")
    ap.add_argument("--seed", type=int, default=1)
    # per-model config (as used in the experiments)
    ap.add_argument("--sg-window", type=int, default=15)
    ap.add_argument("--sg-epochs", type=int, default=5)
    ap.add_argument("--glove-window", type=int, default=30)
    ap.add_argument("--glove-epochs", type=int, default=20)
    ap.add_argument("--cbow-window", type=int, default=10)
    ap.add_argument("--cbow-epochs", type=int, default=7)
    # tools
    ap.add_argument("--warmstart-tool", default=str(HERE / "train_warmstart_word2vec.py"))
    ap.add_argument("--build-tool", default=str(HERE / "build_cooccurrence.py"))
    ap.add_argument("--glove-tool", default=str(HERE / "train_glove.py"))
    ap.add_argument("--workdir", default="timing_tmp")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    py = sys.executable
    wd = Path(a.workdir); wd.mkdir(parents=True, exist_ok=True)
    cooc = wd / "cooc"; cooc.mkdir(exist_ok=True)
    glove_vec = wd / f"glove-d{a.dim}-w{a.glove_window}.vec"

    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        env[k] = str(a.threads)
    if not a.allow_accel:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTORCH_MPS_DISABLE"] = "1"   # honored by recent torch; see note below

    def C(*x): return [str(py)] + [str(t) for t in x]
    sg = C(a.warmstart_tool, "--corpus", a.corpus, "--model","sg","--cold",
           "--dim",a.dim,"--window",a.sg_window,"--epochs",a.sg_epochs,
           "--min-count",a.min_count,"--seed",a.seed,"--out",wd/"sg.vec")
    g_build = C(a.build_tool, "--corpus", a.corpus, "--out-dir", cooc,
                "--no-dynamic-window","--window",a.glove_window,"--min-count",a.min_count)
    g_train = C(a.glove_tool, "--cooc-dir", cooc, "--out", glove_vec,
                "--dim",a.dim,"--epochs",a.glove_epochs,"--seed",a.seed)
    cbow_warm = C(a.warmstart_tool, "--corpus", a.corpus, "--model","cbow",
                  "--glove", glove_vec, "--dim",a.dim,"--window",a.cbow_window,
                  "--epochs",a.cbow_epochs,"--min-count",a.min_count,"--seed",a.seed,
                  "--out",wd/"cbow_warm.vec")
    cbow_cold = C(a.warmstart_tool, "--corpus", a.corpus, "--model","cbow","--cold",
                  "--dim",a.dim,"--window",a.cbow_window,"--epochs",a.cbow_epochs,
                  "--min-count",a.min_count,"--seed",a.seed,"--out",wd/"cbow_cold.vec")

    print("="*72)
    print(f"timing: dim={a.dim} repeats={a.repeats} threads={a.threads} "
          f"accel={'on' if a.allow_accel else 'OFF (CPU)'}")
    print(f"host: {platform.platform()} | python {platform.python_version()} | cpus={os.cpu_count()}")
    print(f"configs: SG(w{a.sg_window},ep{a.sg_epochs})  "
          f"GloVe(w{a.glove_window},ep{a.glove_epochs})  CBOW(w{a.cbow_window},ep{a.cbow_epochs})")
    print("="*72)
    if a.dry_run:
        for name,cmd in [("SG-only",sg),("GloVe:build",g_build),("GloVe:train",g_train),
                         ("CBOW:warm",cbow_warm),("CBOW:cold",cbow_cold)]:
            print(f"  [{name}] " + " ".join(cmd))
        print("\n(dry-run: no execution)"); return

    times = {k: [] for k in ("sg","g_build","g_train","cbow_warm","seq","cbow_cold")}
    logf = open(wd/"stages.log","w")
    for r in range(1, a.repeats+1):
        print(f"\n--- repeat {r}/{a.repeats} ---")
        t = run_timed(sg, env, logf);               times["sg"].append(t);        print(f"  SG-only        {t:8.2f}s")
        tb = run_timed(g_build, env, logf);          times["g_build"].append(tb)
        tt = run_timed(g_train, env, logf);          times["g_train"].append(tt)
        tc = run_timed(cbow_warm, env, logf);        times["cbow_warm"].append(tc)
        times["seq"].append(tb+tt+tc);               print(f"  GloVe→CBOW seq {tb+tt+tc:8.2f}s  (build {tb:.1f} + glove {tt:.1f} + cbow {tc:.1f})")
        tcc = run_timed(cbow_cold, env, logf);       times["cbow_cold"].append(tcc); print(f"  CBOW-cold-only {tcc:8.2f}s")
    med = {k: statistics.median(v) for k,v in times.items() if v}

    print("\n"+"="*72+"\nMEDIAN WALL-CLOCK\n"+"="*72)
    print(f"  SG-only               {med['sg']:8.2f}s")
    print(f"  GloVe→CBOW sequence   {med['seq']:8.2f}s   "
          f"(build {med['g_build']:.1f} + glove {med['g_train']:.1f} + cbow {med['cbow_warm']:.1f})")
    print(f"  CBOW-cold-only (ref)  {med['cbow_cold']:8.2f}s")
    print(f"\n  sequence / SG        = {med['seq']/med['sg']:.2f}×")
    print(f"  CBOW-cold / SG       = {med['cbow_cold']/med['sg']:.2f}×")
    print("\n  Reading: 'sequence/SG < 1' means GloVe→CBOW is cheaper than SG for a")
    print("  model that matches/beats SG on STS/SICK; CBOW-cold/SG is the floor if")
    print("  you want STS/SICK WITHOUT the analogy transfer (no GloVe stage).")
    if not a.allow_accel:
        print("\n  NOTE: forced CPU + threads=%d. If PYTORCH_MPS_DISABLE isn't honored by" % a.threads)
        print("  your torch, check stages.log for the device each torch stage reported")
        print("  and re-run with matched device if needed.")
    logf.close()


if __name__ == "__main__":
    main()
