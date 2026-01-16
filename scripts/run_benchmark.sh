#!/usr/bin/env bash
set -euo pipefail

# Benchmark runner script
# Executa benchmark para uma lista de modelos

# Lista de modelos a executar (igual ao tune_remaining_models.sh)
MODELS=(
    "Random"
    "POP"
    "RPOP"
    "SPOP"
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

# Se passar um modelo específico como argumento, usa apenas ele
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
