#!/bin/bash
# Executar experimentos em paralelo com timestamp compartilhado e agregação única no final

set -e

# Recebe lista de modelos como argumentos, ou usa padrão
if [ $# -eq 0 ]; then
    MODELS=("GRU4Rec" "NARM" "STAMP" "SASRec" "Random" "POP" "RPOP" "SPOP")
else
    MODELS=("$@")
fi

# Gera timestamp único compartilhado por todos os experimentos
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export SHARED_TIMESTAMP="$TIMESTAMP"

ALL_SLICES="1 2 3 4 5"

echo "================================================================================"
echo "                  EXPERIMENTOS PARALELOS - EXECUÇÃO CONTROLADA"
echo "================================================================================"
echo ""
echo "Timestamp compartilhado: $TIMESTAMP"
echo "Modelos: ${MODELS[@]}"
echo "Slices: $ALL_SLICES (executados em paralelo por modelo)"
echo "Diretório de saída: outputs/results/$TIMESTAMP/"
echo ""
echo "================================================================================"
echo ""

# Criar diretório de saída
mkdir -p "outputs/results/$TIMESTAMP/losses"

# Executar todos os modelos em paralelo (cada modelo executa seus próprios slices em paralelo)
pids=()

for model in "${MODELS[@]}"; do
    echo "--------------------------------------------------------------------------------"
    echo "Iniciando $model (todos os slices em paralelo)..."
    echo "--------------------------------------------------------------------------------"
    
    # Roda todos os slices deste modelo em paralelo com timestamp compartilhado
    python src/run_experiments.py \
        --models $model \
        --slices $ALL_SLICES \
        --shared-timestamp "$TIMESTAMP" \
        > "outputs/results/$TIMESTAMP/${model}_execution.log" 2>&1 &
    
    pid=$!
    pids+=($pid)
    echo "  PID $pid: $model"
    echo ""
    
    # Delay para evitar race condition no início
    sleep 3
done

echo ""
echo "================================================================================"
echo "Todos os modelos iniciados. Aguardando conclusão..."
echo "PIDs: ${pids[@]}"
echo "================================================================================"
echo ""

# Aguardar todos os processos
failed=0
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    model=${MODELS[$i]}
    
    if wait $pid; then
        echo "[OK] $model (PID $pid) concluído com sucesso"
    else
        echo "[ERRO] $model (PID $pid) falhou"
        failed=$((failed + 1))
    fi
done

echo ""
echo "================================================================================"

if [ $failed -eq 0 ]; then
    echo "                    TODOS OS EXPERIMENTOS CONCLUÍDOS!"
    echo "================================================================================"
    echo ""
    echo "Executando agregação final..."
    python src/aggregate_results.py \
        --input "outputs/results/$TIMESTAMP" \
        --output "outputs/results/$TIMESTAMP/aggregated_results.csv"
    
    echo ""
    echo "================================================================================"
    echo "RESULTADOS SALVOS EM: outputs/results/$TIMESTAMP/"
    echo "================================================================================"
    echo ""
    echo "Arquivos gerados:"
    ls -lh "outputs/results/$TIMESTAMP/" | grep -v "^d" | awk '{print "  - " $9 " (" $5 ")"}'
    echo ""
else
    echo "                    ALGUNS EXPERIMENTOS FALHARAM!"
    echo "================================================================================"
    echo ""
    echo "Experimentos com falha: $failed de ${#MODELS[@]}"
    echo "Verifique os logs em: outputs/results/$TIMESTAMP/"
    echo ""
fi

echo "Total executado:"
echo "  - Modelos: ${#MODELS[@]}"
echo "  - Slices: 5 por modelo"
echo "  - Total: $((${#MODELS[@]} * 5)) experimentos"
echo ""
echo "================================================================================"
