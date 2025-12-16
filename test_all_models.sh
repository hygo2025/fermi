#!/bin/bash
# Quick test of all models

MODELS=("GRU4Rec" "NARM" "STAMP" "SASRec" "FPMC" "FOSSIL" "Random" "POP" "RPOP" "SPOP")

echo "Testing all 10 models..."
for model in "${MODELS[@]}"; do
    echo "Testing $model..."
    python src/run_experiments.py --models $model --slices 1 --epochs 1 2>&1 | grep -E "Results:" || echo "  ERROR with $model"
done
echo "Done!"
