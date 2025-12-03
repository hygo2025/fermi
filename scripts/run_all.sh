#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                  Fermi - Running All Benchmarks                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

CONFIGS=(
    "src/configs/neural/gru4rec.yaml"
    "src/configs/neural/narm.yaml"
    "src/configs/neural/stamp.yaml"
    "src/configs/neural/srgnn.yaml"
    "src/configs/knn/itemknn.yaml"
    "src/configs/knn/sknn.yaml"
    "src/configs/baselines/pop.yaml"
)

NAMES=(
    "GRU4Rec"
    "NARM"
    "STAMP"
    "SR-GNN"
    "ItemKNN"
    "SKNN"
    "Pop"
)

mkdir -p logs/neural logs/knn logs/baselines

TOTAL=${#CONFIGS[@]}
CURRENT=0

for i in "${!CONFIGS[@]}"; do
    CURRENT=$((i+1))
    CONFIG="${CONFIGS[$i]}"
    NAME="${NAMES[$i]}"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$CURRENT/$TOTAL] Running: $NAME"
    echo "Config: $CONFIG"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    source .venv/bin/activate
    python src/run_recbole.py -c "$CONFIG"
    
    if [ $? -eq 0 ]; then
        echo "✅ $NAME completed successfully"
    else
        echo "❌ $NAME failed"
    fi
    echo ""
done

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                  ✅ All 7 Models Complete!                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved in logs/"
