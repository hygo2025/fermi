#!/bin/bash
# Executar múltiplos experimentos em paralelo na GPU (em lotes de 3)

MODEL=${1:-GRU4Rec}
SLICES=${2:-"1 2 3"}

# Converter slices em array
SLICE_ARRAY=($SLICES)
TOTAL_SLICES=${#SLICE_ARRAY[@]}

echo "================================================================================"
echo "                  EXECUÇÃO PARALELA - GPU RTX 4090"
echo "================================================================================"
echo ""
echo "Modelo: $MODEL"
echo "Slices: $SLICES (total: $TOTAL_SLICES)"
echo "Estratégia: Executar 3 slices por vez"
echo ""
echo "Com batch_size=4096, espera-se ~6-8 GB por slice."
echo ""
echo "================================================================================"
echo ""

# Criar diretório de logs se não existir
mkdir -p outputs/logs

# Função para executar um lote de slices
run_batch() {
    local batch_slices=("$@")
    local pids=()
    
    echo "Iniciando lote de ${#batch_slices[@]} slices..."
    echo ""
    
    for slice in "${batch_slices[@]}"; do
        echo "  → Iniciando $MODEL no slice $slice (background)..."
        
        # Passar shared-timestamp se disponível
        if [ -n "$SHARED_TIMESTAMP" ]; then
            python src/run_experiments.py \
                --models $MODEL \
                --slices $slice \
                --shared-timestamp "$SHARED_TIMESTAMP" \
                > "outputs/logs/${MODEL}_slice${slice}_parallel.log" 2>&1 &
        else
            python src/run_experiments.py \
                --models $MODEL \
                --slices $slice \
                > "outputs/logs/${MODEL}_slice${slice}_parallel.log" 2>&1 &
        fi
        
        pid=$!
        pids+=($pid)
        echo "    PID: $pid"
        echo "    Log: outputs/logs/${MODEL}_slice${slice}_parallel.log"
        echo ""
        
        # Pequeno delay para evitar race condition no RecBole
        sleep 5
    done
    
    # Aguardar todos os slices deste lote
    echo "Aguardando conclusão dos ${#batch_slices[@]} slices..."
    for pid in "${pids[@]}"; do
        wait $pid
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "AVISO: Processo $pid terminou com código $exit_code"
        fi
    done
    
    echo "✓ Lote concluído!"
    echo ""
}

# Executar em lotes de 3
BATCH_SIZE=3
for ((i=0; i<$TOTAL_SLICES; i+=BATCH_SIZE)); do
    # Pegar próximos 3 slices (ou menos se sobrar menos)
    batch=("${SLICE_ARRAY[@]:i:BATCH_SIZE}")
    
    echo "--------------------------------------------------------------------------------"
    echo "LOTE $((i/BATCH_SIZE + 1)): Slices ${batch[@]}"
    echo "--------------------------------------------------------------------------------"
    echo ""
    
    run_batch "${batch[@]}"
done

echo "================================================================================"
echo "✓ TODOS OS SLICES DE $MODEL CONCLUÍDOS!"
echo "================================================================================"
echo ""
echo "Monitorar logs:"
echo "  tail -f outputs/logs/${MODEL}_slice*_parallel.log"
echo ""
