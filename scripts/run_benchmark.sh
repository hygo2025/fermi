#!/usr/bin/env bash
set -euo pipefail


MODELS=(
    "BERT4Rec" #n Rodando
    "TransRec" #n Rodnado
    "FISM"
    "GCSAN"
    "SHAN"
    "SRGNN"
)

MODELS_ALL=(
    "BPR"
    "ItemKNN"
    "Pop"
    "Random"
    "FISM"
    "FOSSIL"
    "FPMC"
    "BERT4Rec"
    "Caser"
    "GCSAN" #n
    "GRU4Rec"
    "NARM"
    "NextItNet"
    "SASRec"
    "SHAN"
    "SRGNN"
    "STAMP"
    "TransRec"
    )

if [[ $# -gt 0 ]]; then
    MODELS=("$1")
fi

echo "[INFO] Running benchmark for ${#MODELS[@]} model(s): ${MODELS[*]}"

for model in "${MODELS[@]}"; do
    echo ""
    echo "[INFO] >>> Starting benchmark for ${model}"
    python src/run_benchmark.py --model "${model}"
    echo "[INFO] <<< Finished ${model}"
    sleep 2
done

echo ""
echo "[INFO] Benchmark complete!"
