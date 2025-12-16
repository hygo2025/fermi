#!/bin/bash
# Script para rodar experimentos em paralelo (2 slices por vez)
# Uso: ./scripts/run_parallel_slices.sh <MODEL> [num_parallel]

set -e

MODEL=${1:-GRU4Rec}
NUM_PARALLEL=${2:-2}
SAVE_CHECKPOINTS=${3:-yes}

echo "================================================"
echo "Rodando $MODEL em paralelo"
echo "Paralelismo: $NUM_PARALLEL slices por vez"
echo "Checkpoints: $SAVE_CHECKPOINTS"
echo "================================================"

# Detectar slices disponíveis
SLICES=($(ls -d outputs/data/recbole/realestate_slice* 2>/dev/null | sed 's/.*slice//'))
TOTAL_SLICES=${#SLICES[@]}

if [ $TOTAL_SLICES -eq 0 ]; then
    echo "ERRO: Nenhum slice encontrado em outputs/data/recbole/"
    exit 1
fi

echo "Total de slices encontrados: $TOTAL_SLICES"
echo "Slices: ${SLICES[@]}"
echo ""

# Construir comando base
CMD="python src/run_experiments.py --models $MODEL"
if [ "$SAVE_CHECKPOINTS" = "yes" ]; then
    CMD="$CMD --save-checkpoints"
fi

# Função para rodar um slice
run_slice() {
    local slice=$1
    local logfile="outputs/logs/${MODEL}_slice${slice}_parallel.log"
    
    echo "[$(date '+%H:%M:%S')] Iniciando slice $slice..."
    
    $CMD --slices $slice > "$logfile" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✓ Slice $slice concluído"
    else
        echo "[$(date '+%H:%M:%S')] ✗ Slice $slice falhou (ver $logfile)"
    fi
}

# Rodar slices em paralelo
pids=()
current=0

for slice in "${SLICES[@]}"; do
    run_slice $slice &
    pids+=($!)
    current=$((current + 1))
    
    # Se atingiu o limite de paralelo, esperar
    if [ $current -ge $NUM_PARALLEL ]; then
        echo ""
        echo "Aguardando conclusão de $NUM_PARALLEL slices..."
        for pid in "${pids[@]}"; do
            wait $pid
        done
        pids=()
        current=0
        echo "Batch concluído. Continuando..."
        echo ""
    fi
done

# Esperar últimos jobs
if [ ${#pids[@]} -gt 0 ]; then
    echo "Aguardando últimos slices..."
    for pid in "${pids[@]}"; do
        wait $pid
    done
fi

echo ""
echo "================================================"
echo "✓ Todos os slices de $MODEL concluídos!"
echo "================================================"
echo ""
echo "Logs individuais em: outputs/logs/${MODEL}_slice*_parallel.log"
echo ""
