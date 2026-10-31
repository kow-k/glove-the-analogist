#!/usr/bin/env bash
# ======================================================================
#  100k scaling campaign — r2 (adds PPMI-SVD as count-based control + warm source)
#  Blocks:
#    A  GloVe + PPMI-SVD scaling / window-extendability (N=1; GloVe d064+d128,
#       PPMI-SVD d128) — both reuse the SAME cooccurrence.npz, so PPMI-SVD runs
#       right after GloVe, before the matrix is deleted.
#    C  Predictive FIXED windows (SG + NS-CBOW, d128, N=3)  <- closes SG-turnover
#    D  Warm NS-CBOW transfer-at-scale: cold vs warm-from-GloVe vs warm-from-
#       PPMI-SVD (g/p 30/40/60, d128, N=3).  Tests whether BOTH count methods
#       hand over analogy, or only GloVe.
#
#  Why PPMI-SVD: it is the other major count-based method (Levy et al. 2015) and
#  the clean control for "is the analogy specialisation GloVe-specific or general
#  to count methods?".  train_ppmi_svd.py defaults to the symmetric W+C recipe
#  (CDS 0.75, Sigma^0.5) — the static sibling of GloVe's W+W~.  Deterministic -> N=1.
#  (Not the temporal "joint learning" of Aida 2022, which is a change-detection
#  method irrelevant to this synchronic comparison.)
#
#  Cost levers: GloVe/PPMI-SVD seed-stable -> N=1.  CBOW uses negative sampling
#  (train_cbow_ns.py, ~0.4x SG, quality-equal) not full softmax.  d128 only for
#  the expensive predictive runs; d064 only for (cheap) GloVe.
#  Idempotent: a stage is skipped if its .json already exists.
#  Storage: cooccurrence .npz deleted after its GloVe+PPMI-SVD trainings;
#  predictive .vec deleted after eval.  GloVe & PPMI-SVD .vec kept (deliverable +
#  warm-start sources).
#
#  Run:  bash run_100k_campaign_r2.sh            (real)
#        DRYRUN=1 bash run_100k_campaign_r2.sh   (print the plan only)
# ======================================================================
set -u

# ----------------------------- CONFIG (edit) --------------------------
CORPUS="corpus-clean4a/sen/senwiki-r100k-clean4a-unmasked.txt"   # 100k corpus
TOOLS="${TOOLS:-src}"                         # dir holding the *.py trainers
EVAL="bash $TOOLS/batch-evaluate-to-json.sh"   # takes a .vec, writes <vec%.vec>.json
OUT="models-100k"                 # output dir for .vec / .json
COOC="cooc-100k"                  # scratch dir for co-occurrence matrices
TAG="sen-r100k"
MIN_COUNT=10                      # higher at 100k: shrinks V & NNZ (report analogy coverage!)
SEEDS_PRED="1 2 3"                # predictive models need seeds
# GloVe
GLOVE_WINS="10 20 30 40 60 70"
GLOVE_DIMS="064 128"
declare -A GLOVE_EP=( [064]=10 [128]=20 )
# PPMI-SVD (count-based control; same windows as GloVe, reuses the cooc matrix)
PPMI_DIMS="128"                   # d128 is what the paper needs; add "064 128" for width-matched count-vs-count
# Predictive fixed-window ladder (Block C) and warm cell (Block D)
PRED_WINS="15 20 25 30 40 50 60 70"
SG_LR=0.025;  SG_EP=05
#NS_LR=0.005;  NS_EP=07;  NS_NEG=5          # matches the validated 30k NS-CBOW config
NS_LR=0.005;  NS_EP=05;  NS_NEG=5          # made shorter for 100k
WARM_SRCS="30 40 60"                          # source windows for Block D (GloVe AND PPMI-SVD)
WARM_TRAIN_WIN=7 #10
# ----------------------------------------------------------------------

run(){ echo "+ $*"; [ "${DRYRUN:-0}" = 1 ] || "$@"; }
evaljson(){ # $1 = vec ; eval only if json missing ; then delete vec unless $2=keep
  local vec="$1" keep="${2:-}"; local js="${vec%.vec}.json"
  if [ -e "$js" ]; then echo "  skip eval (exists): $js"; else
    echo "+ $EVAL $vec"; [ "${DRYRUN:-0}" = 1 ] || $EVAL "$vec"; fi
  [ "$keep" = keep ] || { [ "${DRYRUN:-0}" = 1 ] || rm -f "$vec"; echo "  (rm $vec)"; }
}
mkdir -p "$OUT" "$COOC"

echo "===================== BLOCK A: GloVe + PPMI-SVD (N=1) ====================="
for W in $GLOVE_WINS; do
  cd="$COOC/w$W"
  if [ ! -e "$cd/cooccurrence.npz" ]; then
    run python "$TOOLS/build_cooccurrence.py" --corpus "$CORPUS" --out-dir "$cd" \
        --no-dynamic-window --window "$W" --min-count "$MIN_COUNT"
  else echo "  cooc w$W exists"; fi
  # --- GloVe ---
  for DIM in $GLOVE_DIMS; do
    EP=${GLOVE_EP[$DIM]}
    vec="$OUT/$TAG-glove-plain-d$DIM-w$W-msN-ep$EP-r1.vec"
    if [ -e "${vec%.vec}.json" ]; then echo "  skip glove d$DIM w$W (json exists)"; continue; fi
    run python "$TOOLS/train_glove.py" --cooc-dir "$cd" --out "$vec" \
        --dim "$DIM" --epochs "$EP" --seed 1
    evaljson "$vec" keep          # keep GloVe .vec (deliverable + warm source)
  done
  # --- PPMI-SVD (same cooc matrix; symmetric W+C recipe) ---
  for DIM in $PPMI_DIMS; do
    pv="$OUT/$TAG-ppmisvd-d$DIM-w$W-r1.vec"
    if [ -e "${pv%.vec}.json" ]; then echo "  skip ppmisvd d$DIM w$W (json exists)"; continue; fi
    run python "$TOOLS/train_ppmi_svd.py" --cooc-dir "$cd" --out "$pv" \
        --dim "$DIM" --combine sum --seed 1
    evaljson "$pv" keep           # keep PPMI-SVD .vec (deliverable + warm source)
  done
  # drop the (big) cooc ONLY if every output for this window actually landed,
  # so a failed GloVe/PPMI-SVD run never costs us the matrix.
  keep_cooc=0
  for DIM in $GLOVE_DIMS; do EP=${GLOVE_EP[$DIM]}
    [ -e "$OUT/$TAG-glove-plain-d$DIM-w$W-msN-ep$EP-r1.json" ] || keep_cooc=1; done
  for DIM in $PPMI_DIMS; do
    [ -e "$OUT/$TAG-ppmisvd-d$DIM-w$W-r1.json" ] || keep_cooc=1; done
  if [ "$keep_cooc" = 0 ] && [ "${DRYRUN:-0}" != 1 ]; then
    rm -f "$cd/cooccurrence.npz"; echo "  (rm $cd/cooccurrence.npz)"
  else
    echo "  (keep cooc w$W: outputs incomplete or dry-run)"
  fi
done

echo "============ BLOCK C: SG + NS-CBOW fixed windows (d128, N=3) ============"
for W in $PRED_WINS; do
  for r in $SEEDS_PRED; do
    # Skip-gram (gensim)
    sg="$OUT/$TAG-skgr-cold-d128-w$W-lr$(echo $SG_LR|tr -d .)-ep$SG_EP-r$r.vec"
    [ -e "${sg%.vec}.json" ] && echo "  skip $sg" || {
      run python "$TOOLS/train_warmstart_word2vec.py" --corpus "$CORPUS" --model sg --cold \
          --dim 128 --window "$W" --epochs "$SG_EP" --lr "$SG_LR" --min-count "$MIN_COUNT" --seed "$r" \
          --out "$sg"; evaljson "$sg"; }
    # NS-CBOW
    cb="$OUT/$TAG-cbowNs-cold-d128-w$W-lr$(echo $NS_LR|tr -d .)-ep$NS_EP-r$r.vec"
    [ -e "${cb%.vec}.json" ] && echo "  skip $cb" || {
      run python "$TOOLS/train_cbow_ns.py" --corpus "$CORPUS" --cold \
          --dim 128 --window "$W" --epochs "$NS_EP" --lr "$NS_LR" --negative "$NS_NEG" \
          --min-count "$MIN_COUNT" --seed "$r" --out "$cb"; evaljson "$cb"; }
  done
done

echo "============ BLOCK D: warm NS-CBOW — cold vs GloVe-warm vs PPMI-SVD-warm ============"
# cold baseline at the warm cell (shared reference for both warm sources)
for r in $SEEDS_PRED; do
  cb="$OUT/$TAG-cbowNs-cold-d128-w$WARM_TRAIN_WIN-lr$(echo $NS_LR|tr -d .)-ep$NS_EP-r$r.vec"
  [ -e "${cb%.vec}.json" ] && echo "  skip $cb" || {
    run python "$TOOLS/train_cbow_ns.py" --corpus "$CORPUS" --cold \
        --dim 128 --window "$WARM_TRAIN_WIN" --epochs "$NS_EP" --lr "$NS_LR" --negative "$NS_NEG" \
        --min-count "$MIN_COUNT" --seed "$r" --out "$cb"; evaljson "$cb"; }
done
# warm from GloVe (warm<src>) and from PPMI-SVD (warmPS<src>), same sources
for src in $WARM_SRCS; do
  gsrc="$OUT/$TAG-glove-plain-d128-w$src-msN-ep${GLOVE_EP[128]}-r1.vec"   # from Block A
  psrc="$OUT/$TAG-ppmisvd-d128-w$src-r1.vec"                             # from Block A
  # --- GloVe-warm ---
  if [ ! -e "$gsrc" ]; then echo "  [!] missing GloVe source $gsrc (run Block A first)"; else
    for r in $SEEDS_PRED; do
      cb="$OUT/$TAG-cbowNs-warm$src-d128-w$WARM_TRAIN_WIN-lr$(echo $NS_LR|tr -d .)-ep$NS_EP-r$r.vec"
      [ -e "${cb%.vec}.json" ] && echo "  skip $cb" || {
        run python "$TOOLS/train_cbow_ns.py" --corpus "$CORPUS" --glove "$gsrc" \
            --dim 128 --window "$WARM_TRAIN_WIN" --epochs "$NS_EP" --lr "$NS_LR" \
            --negative "$NS_NEG" \
            --min-count "$MIN_COUNT" --seed "$r" --out "$cb"; evaljson "$cb"; }
    done
  fi
  # --- PPMI-SVD-warm ---
  if [ ! -e "$psrc" ]; then echo "  [!] missing PPMI-SVD source $psrc (run Block A first)"; else
    for r in $SEEDS_PRED; do
      cb="$OUT/$TAG-cbowNs-warmPS$src-d128-w$WARM_TRAIN_WIN-lr$(echo $NS_LR|tr -d .)-ep$NS_EP-r$r.vec"
      [ -e "${cb%.vec}.json" ] && echo "  skip $cb" || {
        run python "$TOOLS/train_cbow_ns.py" --corpus "$CORPUS" --glove "$psrc" \
            --dim 128 --window "$WARM_TRAIN_WIN" --epochs "$NS_EP" --lr "$NS_LR" \
            --negative "$NS_NEG" \
            --min-count "$MIN_COUNT" --seed "$r" --out "$cb"; evaljson "$cb"; }
    done
  fi
done

echo "DONE. JSONs in $OUT ; aggregate with aggregate_seeds.py, read GloVe/PPMI-SVD curves with associate_window.py."
