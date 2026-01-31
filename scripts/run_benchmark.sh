#!/usr/bin/env bash
set -euo pipefail

MODELS=(
    "BPR"
    "ItemKNN"
    "Pop"
    "Random"
    "FOSSIL"
    "FPMC"
    "BERT4Rec"
    "Caser"
    "GCSAN"
    "GRU4Rec"
    "NARM"
    "SASRec"
    "SRGNN"
    "STAMP"
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
