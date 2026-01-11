#!/usr/bin/env bash
set -euo pipefail

# Remaining models after overnight run
REMAINING_MODELS=("FOSSIL")

MAX_EVALS=${MAX_EVALS:-150}
ALGO=${ALGO:-bayes}
COOLDOWN=${COOLDOWN:-60}
DATASET=${DATASET:-}

for model in "${REMAINING_MODELS[@]}"; do
  echo "[INFO] >>> Tuning ${model} (MAX_EVALS=${MAX_EVALS}, ALGO=${ALGO}, COOLDOWN=${COOLDOWN})"
  if [[ -n "${DATASET}" ]]; then
    make tune MODEL="${model}" MAX_EVALS="${MAX_EVALS}" ALGO="${ALGO}" COOLDOWN="${COOLDOWN}" DATASET="${DATASET}"
  else
    make tune MODEL="${model}" MAX_EVALS="${MAX_EVALS}" ALGO="${ALGO}" COOLDOWN="${COOLDOWN}"
  fi
  echo "[INFO] <<< Finished ${model}"
  echo
  sleep 5
done
