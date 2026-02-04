#!/usr/bin/env bash
set -euo pipefail


MODELS=(
    "TransRec"
    "SRGNN"
    "GCSAN"
    "FISM"
    "FPMC"
    "BERT4Rec"
)

MODELS_ALL=(
    "BPR"
    "ItemKNN"
    "Pop"
    "Random"
    "FISM" #n
    "FOSSIL" #n -no cafe
    "FPMC" #n -no cafe
    "BERT4Rec" #n -no cafe
    "Caser"
    "GCSAN" #n
    "GRU4Rec"
    "NARM"
    "NextItNet" #n
    "SASRec"
    "SHAN"
    "SRGNN" #n
    "STAMP"
    "TransRec" #n
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
