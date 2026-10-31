#!/usr/bin/env bash
# Warm-vs-cold grid at 10k/30k (the seeded 10k/30k set in the paper).
# Run from the repo root; set SIZE below to r010k or r030k.
TOOLS="${TOOLS:-src}"
#GLOVE=<a trained GloVe .vec, dim must match --dim>

CAUTIOUS=true

DIM_BASE=32
EPOCH_BASE=05
MIN_COUNT=05
MAX_SPAN=None

LANG=sen
CLASS=clean4a
SIZE=r010k
CORPUS=corpus-${CLASS}/${LANG}/${LANG}wiki-${SIZE}-${CLASS}-unmasked.txt
MODEL_DIR=models-GloVe/${LANG}/${CLASS}/${SIZE}

echo "Started training..."
for DIM in 064 128; do
  echo "  DIM: $DIM"
  if [[ ${DIM} == 064 ]]; then
    GLOVE_EPOCHS=10
  else
    GLOVE_EPOCHS=20
  fi

  ## Loop over Glove window sizes
  for gW in 10 20 30; do
    GLOVE=${MODEL_DIR}/${LANG}-${SIZE}-glove-d${DIM}-plain-w${gW}-msNone-ep${GLOVE_EPOCHS}-r1.vec
    echo "  Source GloVe: ${GLOVE}"
    if [[ -e ${GLOVE} && ${CAUTIOUS} ]]; then
      echo "  Found ${GLOVE}"
    else
      echo "  ##Error: failed to find ${GLOVE}"
    fi
    
    ## Loop over architectures
    for M in cbow sg; do
      ## Skipgram
      if [[ ${M} == 'sg' ]]; then
        M_CODE=skgr
        LR=0.025
        LR_CODE=$(echo "$LR" | tr -d ".")
        EPOCHS=05
      ## CBoW
      else
        M_CODE=${M}
        LR=0.015 #0.005 # Gensim needs higher alphas?
        LR_CODE=$(echo "$LR" | tr -d ".")
        EPOCHS=20 # Gensim needs longer epochs?
      fi
      ## Loop over SEED
      for r in 1 2 3; do
        ## Loop over target windows
        for W in 05 10 15; do
          ## Warm
          WARM=${LANG}-${SIZE}-${M_CODE}-warm-d${DIM}-w${W}-lr${LR_CODE}-ep${EPOCHS}-r${r}.vec
          echo "  WARM: ${WARM}"
          if [[ -e ${MODEL_DIR}/${WARM} && ${CAUTIOUS} == true ]]; then
            echo "  Target file exists: Skipped creating ${WARM}"
          else
            python "$TOOLS/train_warmstart_word2vec.py" --corpus ${CORPUS} \
              --model ${M} --dim ${DIM} --lr ${LR} --window ${W} \
              --epochs ${EPOCHS} --min-count ${MIN_COUNT} --seed ${r} --glove ${GLOVE} \
              --out ${MODEL_DIR}/${WARM}
            #
            bash "$TOOLS/batch-evaluate-to-json.sh" ${MODEL_DIR}/${WARM}
          fi
          ## Cold
          COLD=${LANG}-${SIZE}-${M_CODE}-cold-d${DIM}-w${W}-lr${LR_CODE}-ep${EPOCHS}-r${r}.vec
          echo "  COLD: ${COLD}"
          if [[ -e $MODEL_DIR/${COLD} && ${CAUTIOUS} == true ]]; then
            echo "  Target file exists: Skipped creating ${COLD}"
          else
            python "$TOOLS/train_warmstart_word2vec.py" --corpus ${CORPUS} \
              --cold --model ${M} --dim ${DIM} --lr ${LR} --window ${W} \
              --epochs ${EPOCHS} --min-count ${MIN_COUNT} --seed ${r} \
              --out ${MODEL_DIR}/${COLD}
              #
              bash "$TOOLS/batch-evaluate-to-json.sh" ${MODEL_DIR}/${COLD}
          fi
          echo ""
        done # W
      done # SEED
    done # Model M
  done # gW
done # DIM

echo "All GloVe-initialized Skip-gram/CBoW trainings ended"
# eval all, then: aggregate_seeds.py --pair <warm> <cold>