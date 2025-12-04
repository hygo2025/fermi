#!/bin/bash
# Executar todos os experimentos com paralelização controlada
# 3 slices por vez, aguarda conclusão antes de iniciar próximo batch

set -e

MODELS=("GRU4Rec" "NARM" "STAMP" "SASRec")
BATCH1="1 2 3"
BATCH2="4 5"

echo "================================================================================"
echo "                  EXPERIMENTOS PARALELOS - EXECUÇÃO CONTROLADA"
echo "================================================================================"
echo ""
echo "Estratégia: 3 slices simultâneos por modelo"
echo "Modelos: ${MODELS[@]}"
echo "Batch 1: Slices $BATCH1"
echo "Batch 2: Slices $BATCH2"
echo ""
echo "================================================================================"
echo ""

for model in "${MODELS[@]}"; do
    echo "--------------------------------------------------------------------------------"
    echo "Modelo: $model"
    echo "--------------------------------------------------------------------------------"
    echo ""
    
    # Batch 1: Slices 1, 2, 3
    echo "[Batch 1] Iniciando slices $BATCH1 em paralelo..."
    pids=()
    
    for slice in $BATCH1; do
        echo "  - Iniciando $model slice $slice..."
        python src/run_experiments.py \
            --models $model \
            --slices $slice \
            > "results/logs/${model}_slice${slice}_parallel.log" 2>&1 &
        
        pid=$!
        pids+=($pid)
        echo "    PID: $pid"
        
        # Delay para evitar race condition
        sleep 5
    done
    
    echo ""
    echo "[Batch 1] Aguardando conclusão dos slices $BATCH1..."
    for pid in "${pids[@]}"; do
        wait $pid
        echo "  - Processo $pid concluído"
    done
    
    echo ""
    echo "[Batch 1] Concluído! Slices $BATCH1 finalizados."
    echo ""
    
    # Batch 2: Slices 4, 5
    echo "[Batch 2] Iniciando slices $BATCH2 em paralelo..."
    pids=()
    
    for slice in $BATCH2; do
        echo "  - Iniciando $model slice $slice..."
        python src/run_experiments.py \
            --models $model \
            --slices $slice \
            > "results/logs/${model}_slice${slice}_parallel.log" 2>&1 &
        
        pid=$!
        pids+=($pid)
        echo "    PID: $pid"
        
        # Delay para evitar race condition
        sleep 5
    done
    
    echo ""
    echo "[Batch 2] Aguardando conclusão dos slices $BATCH2..."
    for pid in "${pids[@]}"; do
        wait $pid
        echo "  - Processo $pid concluído"
    done
    
    echo ""
    echo "[Batch 2] Concluído! Slices $BATCH2 finalizados."
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "Modelo $model: CONCLUÍDO (todos os 5 slices)"
    echo "--------------------------------------------------------------------------------"
    echo ""
    
    # Pausa entre modelos para GPU esfriar
    if [ "$model" != "SASRec" ]; then
        echo "Pausa de 60s para resfriamento da GPU antes do próximo modelo..."
        sleep 60
        echo ""
    fi
done

echo "================================================================================"
echo "                        TODOS OS EXPERIMENTOS CONCLUÍDOS!"
echo "================================================================================"
echo ""
echo "Total executado:"
echo "  - Modelos: ${#MODELS[@]}"
echo "  - Slices: 5 por modelo"
echo "  - Total: $((${#MODELS[@]} * 5)) experimentos"
echo ""
echo "Próximos passos:"
echo "  1. Verificar logs: ls -lh results/logs/"
echo "  2. Agregar resultados: make aggregate-results"
echo "  3. Ver resultados: cat results/aggregated_results.csv"
echo ""
echo "================================================================================"
