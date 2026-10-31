# glove-the-analogist

Code and result tables for the paper

> **GloVe is an Analogy specialist: Insights from a systematic comparison among CBoW, Skip-gram, GloVe and PPMI-SVD**
> Kow Kuroda (2026). *Preprint link to be added.*

The paper compares four classic word-embedding objectives (GloVe, CBOW,
Skip-gram with negative sampling, and PPMI-SVD) on Simple English Wikipedia at
three corpus sizes (10k, 30k and 100k articles), over intrinsic benchmarks that
separate sentence composition, lexical similarity and relational/analogical
structure.

## Main findings

- **Four-way specialization.** CBOW leads on composition (STS, SICK), Skip-gram
  on lexical similarity (SimLex-999), GloVe on analogy, and PPMI-SVD on
  relatedness and outlier structure (WordSim-353, outlier detection, OPP).
- **Analogy depends on the objective, not on counting.** GloVe and PPMI-SVD
  factorize the same co-occurrence counts, yet GloVe is the strongest analogy
  model (0.397 at 100k) while PPMI-SVD is among the weakest (0.178).
- **Distinct window-responses for analogy.** GloVe climbs to the highest
  plateau, Skip-gram extends to a lower one, CBOW peaks early and turns over,
  and PPMI-SVD stays flat.
- **Corpus size is the primary driver** of GloVe's analogy (about 0.10, 0.222
  and 0.397 at 10k, 30k and 100k); window size is secondary.
- **Warm-starting does not transfer GloVe's analogy.** Initializing CBOW or
  Skip-gram from GloVe leaves warm and cold models practically
  indistinguishable.
- **CBOW overshoots and erodes.** Against optimizer updates, CBOW's relational
  scores rise to an early peak and then decline, at a roughly fixed update count
  regardless of corpus size; composition keeps saturating. Skip-gram rises
  monotonically.

## Repository layout

```
src/        trainers, evaluator, analysis scripts (all run from the repo root)
scripts/    driver scripts for the experiment campaigns
plots/      figure scripts
results/    aggregated result tables behind the paper's tables and figures
data/       benchmark files go here (not distributed; see data/README.md)
```

| File | Purpose |
|---|---|
| `src/build_cooccurrence.py` | Sparse word-context co-occurrence matrix from a corpus (fixed or dynamic window). Shared input for GloVe and PPMI-SVD. |
| `src/train_glove.py` | GloVe (PyTorch, Adagrad); final vectors W + W~, x_max derived from the counts. |
| `src/train_ppmi_svd.py` | PPMI-SVD (context-distribution smoothing 0.75, U·Σ^0.5, W + C), from the same matrix. |
| `src/train_warmstart_word2vec.py` | Integrated trainer: `--model sg` (gensim SGNS), `cbow` (full-softmax CBOW, PyTorch + Adam), `cbowns` (negative-sampling CBOW). Cold (`--cold`) or warm-started from GloVe (`--glove`). |
| `src/train_cbow_ns.py` | Stand-alone negative-sampling CBOW (used by the 100k campaign and trajectory runs). |
| `src/train_warmstart_cbow.py` | Stand-alone full-softmax CBOW. |
| `src/evaluate_embeddings.py` | Evaluator for all reported benchmarks; writes one JSON per model. |
| `src/batch-evaluate-to-json.sh` | Evaluation wrapper the drivers call: `x.vec` → `x.json`. |
| `src/analyze_warm_transfer.py` | Warm-vs-cold sign tests over the matched grid. |
| `src/aggregate_seeds.py` | Mean ± s.d. over seeds, and paired comparisons. |
| `src/cost_compare.py`, `src/time_compare.py` | Implementation-independent operation counts, and measured wall-clock. |
| `scripts/run_warmstart_10k30k.sh` | Warm/cold CBOW and Skip-gram grid at 10k and 30k. |
| `scripts/run_100k_campaign.sh` | 100k campaign: GloVe and PPMI-SVD window ladder, predictive window ladder, warm-start cells. |
| `scripts/run_trajectory.sh` | Multi-seed training trajectories with per-epoch checkpoints. |
| `plots/make_paper_plots.py` | Figures 2–3 (30k warm-start view, corpus-size trends) from `results/results_d128.txt`. |
| `plots/fig_window_100k.py` | Figure 1 (100k window ladder, four methods) from `results/results_100k.txt`. |
| `plots/plot_epoch_corpus.py`, `plot_epoch_methods.py`, `plot_epoch_trajectory.py` | Trajectory figures (Figure 4) from checkpoint JSONs. |

## Installation

Python 3.9 or later.

```bash
pip install -r requirements.txt
```

The CBOW trainers use PyTorch and run on CPU, CUDA or Apple MPS.

## Data

**Corpus.** Simple English Wikipedia, as pre-tokenized plain text with one
sentence per line. The corpus and its preprocessing are not part of this
repository; any corpus in that format works.

**Benchmarks.** Put the benchmark files in `data/` (or point to them with
environment variables). Sources and formats are listed in
[`data/README.md`](data/README.md). The benchmarks are not redistributed here.

## Reproducing the experiments

All commands run from the repository root. Every trainer writes text-format
vectors (`word v1 ... vd`), and the evaluator writes a JSON file with the
scores beside them.

**One model of each kind**, at window 20 and d = 128:

```bash
# shared co-occurrence matrix, then the two count-based models
python src/build_cooccurrence.py --corpus corpus.txt --out-dir cooc-w20 \
    --no-dynamic-window --window 20 --min-count 10
python src/train_glove.py    --cooc-dir cooc-w20 --dim 128 --epochs 20 --seed 1 --out glove-w20.vec
python src/train_ppmi_svd.py --cooc-dir cooc-w20 --dim 128 --out ppmisvd-w20.vec

# predictive models, cold start
python src/train_warmstart_word2vec.py --corpus corpus.txt --model sg --cold \
    --dim 128 --window 20 --epochs 5 --lr 0.025 --min-count 10 --seed 1 --out sg-w20.vec
python src/train_cbow_ns.py --corpus corpus.txt --cold \
    --dim 128 --window 20 --epochs 5 --lr 0.005 --negative 5 --min-count 10 --seed 1 --out cbowns-w20.vec

# the same CBOW, warm-started from GloVe
python src/train_cbow_ns.py --corpus corpus.txt --glove glove-w20.vec \
    --dim 128 --window 20 --epochs 5 --lr 0.005 --negative 5 --min-count 10 --seed 1 --out cbowns-warm-w20.vec

# evaluate (writes glove-w20.json, ...)
bash src/batch-evaluate-to-json.sh glove-w20.vec ppmisvd-w20.vec sg-w20.vec cbowns-w20.vec cbowns-warm-w20.vec
```

**Full campaigns.** Edit the configuration block at the top of each driver
(corpus path, output directory), then run it. `DRYRUN=1` prints the plan without
running anything. Stages whose JSON already exists are skipped, so a run can be
resumed.

```bash
DRYRUN=1 bash scripts/run_100k_campaign.sh
bash scripts/run_100k_campaign.sh
```

**Training trajectories.** Add `--save-each-epoch`, or `--save-every-steps N`
for checkpoints every N optimizer updates (the update axis of Figure 4), to the
CBOW trainers; evaluate each checkpoint, then plot:

```bash
python plots/plot_epoch_corpus.py 50k=traj-50k 100k=traj-100k 300k=traj-300k \
    --glob '*cbowNs*st*-r*.json' --xaxis step \
    --metrics STS-A SICK-A Outlier OPP Analogy WS-353 SL-999 --ncols 4 \
    --out fig_cbow_updates_corpus.pdf
```

**Analysis.**

```bash
python src/analyze_warm_transfer.py --dir models-100k --by-source   # warm-vs-cold sign tests
python src/aggregate_seeds.py --results runs.txt                    # mean ± s.d. over seeds
python src/cost_compare.py --vocab-size 20400 --tokens 1490000 --nnz 12100000   # 30k cost ratios (0.36x / 0.40x / 0.77x)
```

**Paper figures 1–3** from the bundled tables:

```bash
python plots/fig_window_100k.py
python plots/make_paper_plots.py
```

## Results

`results/results_100k.txt` (100k: GloVe, PPMI-SVD, Skip-gram and NS-CBOW over
the window ladder) and `results/results_d128.txt` (10k and 30k, d = 128, cold and
warm) are tab-separated, one row per configuration, with the scores of the seven
metrics: means over seeds at 100k, mean ± s.d. at 10k/30k (`N` is the number of
seeds). Model names encode the
configuration, e.g. `r100k-cbowNs-cold-d128-w10-lr0005-ep05` = 100k corpus,
negative-sampling CBOW, cold start, d = 128, window 10, learning rate 0.005,
5 epochs.

## Evaluation

| Metric | JSON key | Score |
|---|---|---|
| STS-A | `sts_plain` | Spearman ρ, cosine of averaged word vectors |
| SICK-A | `sick_plain` | as STS-A |
| SL-999 | `simlex` | Spearman ρ over in-vocabulary pairs |
| WS-353 | `wordsim` | Spearman ρ over in-vocabulary pairs |
| Analogy | `analogy` | 3CosAdd accuracy over full-vocabulary search |
| Outlier | `outlier` | outlier-detection accuracy (compactness score) |
| OPP | `outlier_opp` | Outlier Position Percentage |

`src/evaluate_embeddings.py` is a reduced version of the author's evaluator for
pattern-lattice models. It keeps only the plain word-vector path used for every
number in the paper, with the same loaders, scoring and JSON keys, and gives
identical scores on plain word vectors.

## Citation

```bibtex
@misc{kuroda2026glove,
  author = {Kuroda, Kow},
  title  = {{GloVe} is an Analogy specialist: Insights from a systematic
            comparison among {CBoW}, Skip-gram, {GloVe} and {PPMI-SVD}},
  year   = {2026},
  note   = {Preprint}
}
```

## License

MIT; see [LICENSE](LICENSE).
