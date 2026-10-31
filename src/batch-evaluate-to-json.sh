#!/usr/bin/env bash
# Evaluate word-vector file(s) and write <name>.json next to each <name>.vec.
# This is the evaluation step the driver scripts call ($EVAL <file.vec>).
#
#   bash src/batch-evaluate-to-json.sh models/x.vec [more.vec ...]
#
# Benchmark files are taken from $EVAL_DATA (default: data/ at the repo root);
# each can be overridden individually. See data/README.md for sources/formats.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${EVAL_DATA:-$HERE/../data}"
STS="${STS:-$DATA/sts-benchmark.csv}"
SICK="${SICK:-$DATA/SICK.txt}"
SIMLEX="${SIMLEX:-$DATA/SimLex-999.txt}"
WORDSIM="${WORDSIM:-$DATA/wordsim353-combined.csv}"
ANALOGY="${ANALOGY:-$DATA/questions-words.txt}"
OUTLIER="${OUTLIER:-$DATA/outlier-detection.tsv}"
PYTHON="${PYTHON:-python}"

# Fail loudly: a missing file would otherwise fall back to the evaluator's small
# built-in samples and silently give different numbers.
for f in "$STS" "$SICK" "$SIMLEX" "$WORDSIM" "$ANALOGY" "$OUTLIER"; do
  [ -e "$f" ] || { echo "missing benchmark file: $f (see data/README.md)" >&2; exit 1; }
done
[ $# -ge 1 ] || { echo "usage: $0 file.vec [file.vec ...]" >&2; exit 2; }

for vec in "$@"; do
  "$PYTHON" "$HERE/evaluate_embeddings.py" "$vec" \
      --sts "$STS" --sick "$SICK" --simlex "$SIMLEX" --wordsim "$WORDSIM" \
      --analogy "$ANALOGY" --outlier "$OUTLIER" \
      --output-json "$(dirname "$vec")/{model}.json"
done
