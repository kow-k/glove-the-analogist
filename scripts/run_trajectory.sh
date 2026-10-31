#!/usr/bin/env bash
# ======================================================================
#  Epoch-trajectory sweep for CBOW vs Skip-gram, multi-seed.
#  For each method x seed: train ONCE with --save-each-epoch (checkpoints
#  ...-ep01-r<seed>.vec ... -epNN-r<seed>.vec), evaluate every checkpoint,
#  then delete the (big) .vec keeping only the .json. The plotters pool the
#  seeds into a mean +- sigma band.
#
#  Plot:
#    python plot_epoch_methods.py \
#      "CBOW=$OUT=*cbowNs-cold-d128-w${W}-*ep*-r*.json" \
#      "SG=$OUT=*skgr-cold-d128-w${W}-*ep*-r*.json" \
#      --metrics Analogy STS-A WS-353 --band std --out fig_epoch_methods.pdf
#
#  Run:  bash run_trajectory.sh            (real)
#        DRYRUN=1 bash run_trajectory.sh   (print the plan only)
# ======================================================================
set -u

# ----------------------------- CONFIG (edit) --------------------------
CORPUS="corpus-clean4a/sen/senwiki-r100k-clean4a-unmasked.txt"
TOOLS="${TOOLS:-src}"
EVAL="bash $TOOLS/batch-evaluate-to-json.sh"     # takes a .vec, writes <vec%.vec>.json
OUT="models-traj"
TAG="sen-r100k"
W=10                                       # training window (match across methods)
MIN_COUNT=10
SEEDS="1 2 3"
CB_EP=15                                   # CBOW epochs to trace
SG_EP=15                                   # SG epochs to trace (match for a fair x-axis)
NS_LR=0.005; NS_NEG=5                       # CBOW-NS config
SG_LR=0.025                                 # Skip-gram config
DO_CBOW=1; DO_SG=1
# ----------------------------------------------------------------------

run(){ echo "+ $*"; [ "${DRYRUN:-0}" = 1 ] || "$@"; }
# eval a checkpoint if its json is missing, then drop the .vec (keep json)
evaljson(){ local vec="$1" js="${1%.vec}.json"
  if [ -e "$js" ]; then echo "  skip eval (exists): $js"
  else echo "+ $EVAL $vec"; [ "${DRYRUN:-0}" = 1 ] || $EVAL "$vec"; fi
  [ "${DRYRUN:-0}" = 1 ] || rm -f "$vec"; }
mkdir -p "$OUT"

echo "===== CBOW-NS + Skip-gram epoch trajectories (w$W, seeds: $SEEDS) ====="
for r in $SEEDS; do

  if [ "$DO_CBOW" = 1 ]; then
    lrtag=$(echo $NS_LR | tr -d .)
    base="$OUT/$TAG-cbowNs-cold-d128-w$W-lr$lrtag-ep$CB_EP-r$r.vec"
    last="$OUT/$TAG-cbowNs-cold-d128-w$W-lr$lrtag-ep$(printf %02d $CB_EP)-r$r.json"
    if [ -e "$last" ]; then echo "  skip CBOW train seed $r (last-epoch json exists)"; else
      run python "$TOOLS/train_cbow_ns.py" --corpus "$CORPUS" --cold \
          --dim 128 --window "$W" --epochs "$CB_EP" --negative "$NS_NEG" \
          --lr "$NS_LR" --min-count "$MIN_COUNT" --seed "$r" \
          --save-each-epoch --out "$base"; fi
    for e in $(seq 1 $CB_EP); do
      evaljson "$OUT/$TAG-cbowNs-cold-d128-w$W-lr$lrtag-ep$(printf %02d $e)-r$r.vec"
    done
  fi

  if [ "$DO_SG" = 1 ]; then
    lrtag=$(echo $SG_LR | tr -d .)
    base="$OUT/$TAG-skgr-cold-d128-w$W-lr$lrtag-ep$SG_EP-r$r.vec"
    last="$OUT/$TAG-skgr-cold-d128-w$W-lr$lrtag-ep$(printf %02d $SG_EP)-r$r.json"
    if [ -e "$last" ]; then echo "  skip SG train seed $r (last-epoch json exists)"; else
      run python "$TOOLS/train_warmstart_word2vec.py" --corpus "$CORPUS" --model sg --cold \
          --dim 128 --window "$W" --epochs "$SG_EP" --lr "$SG_LR" \
          --min-count "$MIN_COUNT" --seed "$r" \
          --save-each-epoch --out "$base"; fi
    for e in $(seq 1 $SG_EP); do
      evaljson "$OUT/$TAG-skgr-cold-d128-w$W-lr$lrtag-ep$(printf %02d $e)-r$r.vec"
    done
  fi

done
echo "DONE. JSONs in $OUT ; plot with plot_epoch_methods.py (see header)."
