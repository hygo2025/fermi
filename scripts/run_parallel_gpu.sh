#!/bin/bash
# Script para rodar experimentos em paralelo com gerenciamento de GPU
# Usa GPU 0 e 1 alternadamente (se tiver 2 GPUs) ou mesma GPU

set -e

MODEL=${1:-GRU4Rec}
SAVE_CHECKPOINTS=${2:-yes}

echo "================================================"
echo "Rodando $MODEL em 2 GPUs/processos paralelos"
echo "Checkpoints: $SAVE_CHECKPOINTS"
echo "================================================"

# Detectar slices disponíveis
SLICES=($(ls -d outputs/data/recbole/realestate_slice* 2>/dev/null | sed 's/.*slice//' | sort -n))
TOTAL_SLICES=${#SLICES[@]}

if [ $TOTAL_SLICES -eq 0 ]; then
    echo "ERRO: Nenhum slice encontrado em outputs/data/recbole/"
    exit 1
fi

echo "Total de slices: $TOTAL_SLICES"
echo "Slices: ${SLICES[@]}"
echo ""

# Detectar GPUs disponíveis
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
echo "GPUs disponíveis: $NUM_GPUS"

if [ $NUM_GPUS -ge 2 ]; then
    echo "Usando GPU 0 e GPU 1 para processos paralelos"
    GPU_IDS=(0 1)
else
    echo "Apenas 1 GPU disponível, rodando 2 processos na mesma GPU"
    GPU_IDS=(0 0)
fi
echo ""

# Gerar timestamp compartilhado para todos os slices
SHARED_TIMESTAMP=$(date '+%b-%d-%Y_%H-%M-%S')
echo "Timestamp compartilhado: $SHARED_TIMESTAMP"

# Construir comando base
CMD="python src/run_experiments.py --models $MODEL --shared-timestamp $SHARED_TIMESTAMP"
if [ "$SAVE_CHECKPOINTS" = "yes" ]; then
    CMD="$CMD --save-checkpoints"
fi

# Criar diretório de logs
mkdir -p outputs/logs

# Função para rodar um slice em GPU específica
run_slice_on_gpu() {
    local slice=$1
    local gpu=$2
    local logfile="outputs/logs/${MODEL}_slice${slice}_gpu${gpu}.log"
    
    echo "[$(date '+%H:%M:%S')] GPU $gpu: Iniciando slice $slice..."
    
    # Forçar GPU específica
    CUDA_VISIBLE_DEVICES=$gpu $CMD --slices $slice > "$logfile" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] GPU $gpu: OK - Slice $slice concluído"
    else
        echo "[$(date '+%H:%M:%S')] GPU $gpu: ERRO - Slice $slice falhou (código: $exit_code)"
        echo "                          Ver log: $logfile"
    fi
    
    return $exit_code
}

# Rodar em batches de 2
total_processed=0
failed_slices=()

for ((i=0; i<$TOTAL_SLICES; i+=2)); do
    slice1=${SLICES[$i]}
    slice2=${SLICES[$((i+1))]}
    
    echo ""
    echo "─────────────────────────────────────────────────"
    echo "Batch $((i/2 + 1)): Slices $slice1"
    if [ ! -z "$slice2" ]; then
        echo "            e     $slice2"
    fi
    echo "─────────────────────────────────────────────────"
    
    # Rodar slice1 em GPU 0
    run_slice_on_gpu $slice1 ${GPU_IDS[0]} &
    pid1=$!
    total_processed=$((total_processed + 1))
    
    # Rodar slice2 em GPU 1 (se existir)
    if [ ! -z "$slice2" ]; then
        run_slice_on_gpu $slice2 ${GPU_IDS[1]} &
        pid2=$!
        total_processed=$((total_processed + 1))
        
        # Esperar ambos
        wait $pid1 || failed_slices+=($slice1)
        wait $pid2 || failed_slices+=($slice2)
    else
        # Apenas 1 slice restante
        wait $pid1 || failed_slices+=($slice1)
    fi
    
    echo ""
    echo "Batch concluído. Processados: $total_processed/$TOTAL_SLICES slices"
done

echo ""
echo "================================================"
echo "Processamento concluído!"
echo "================================================"
echo "Total de slices: $TOTAL_SLICES"
echo "Processados: $total_processed"
echo "Falhas: ${#failed_slices[@]}"

if [ ${#failed_slices[@]} -gt 0 ]; then
    echo ""
    echo "AVISO: Slices com falha: ${failed_slices[@]}"
    echo "Ver logs em: outputs/logs/${MODEL}_slice*_gpu*.log"
    exit 1
else
    echo ""
    echo "Todos os slices foram processados com sucesso!"
fi

echo ""
echo "Logs individuais: outputs/logs/${MODEL}_slice*_gpu*.log"
echo "Checkpoints: outputs/saved/${MODEL}-*.pth"
echo ""
