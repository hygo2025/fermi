#!/usr/bin/env bash
set -euo pipefail

MODELS=(
    "STAMP"
    "BPR"
    "FPMC"
    "Caser"
    "SASRec"
    "ItemKNN"
    "Pop"
    #"GRU4Rec"
    #"NARM"
    #"FOSSIL"
    #"BERT4Rec"
    #"SRGNN"
    #"GCSAN"
    #"Random"
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
