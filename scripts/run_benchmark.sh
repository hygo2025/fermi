#!/usr/bin/env bash
set -euo pipefail


DEFAULT_MODELS=(
    "BERT4Rec"
    "TransRec"
    "GCSAN"
    "SRGNN"
)

MODELS_ALL=(
    "BPR"
    "ItemKNN"
    "Pop"
    "Random"
    "FOSSIL"
    "LightGCN"
    "FPMC"
    "BERT4Rec"
    "Caser"
    "GRU4Rec"
    "NARM"
    "NextItNet"
    "SASRec"
    "TransRec"
    "GCSAN"
    "SRGNN"
    )

if [[ $# -gt 0 ]]; then
    MODELS_LIST=("$1")
elif [[ -n "${MODELS:-}" ]]; then
    IFS=' ' read -r -a MODELS_LIST <<< "$MODELS"
else
    MODELS_LIST=("${DEFAULT_MODELS[@]}")
fi

echo "[INFO] Running benchmark for ${#MODELS_LIST[@]} model(s): ${MODELS_LIST[*]}"
RESUME_ARG=""
if [[ -n "${RESUME:-}" ]]; then
    if [[ ${#MODELS_LIST[@]} -eq 1 ]]; then
        RESUME_ARG="--resume ${RESUME}"
    else
        echo "[WARN] RESUME provided but multiple models selected; ignoring RESUME."
    fi
fi

for model in "${MODELS_LIST[@]}"; do
    echo ""
    echo "[INFO] >>> Starting benchmark for ${model}"
    python src/run_benchmark.py --model "${model}" ${RESUME_ARG}
    echo "[INFO] <<< Finished ${model}"
    sleep 2
done

echo ""
echo "[INFO] Benchmark complete!"
