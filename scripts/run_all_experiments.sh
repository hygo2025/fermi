#!/bin/bash
# Executar todos os experimentos usando run_parallel.sh

set -e

MODELS=("GRU4Rec" "NARM" "STAMP" "SASRec" "Random" "POP" "RPOP" "SPOP")
ALL_SLICES="1 2 3 4 5"

# Gera timestamp único compartilhado
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export SHARED_TIMESTAMP="$TIMESTAMP"

echo "================================================================================"
echo "                  EXECUTAR TODOS OS EXPERIMENTOS"
echo "================================================================================"
echo ""
echo "Timestamp compartilhado: $TIMESTAMP"
echo "Modelos: ${MODELS[@]}"
echo "Slices por modelo: $ALL_SLICES (executados em paralelo)"
echo "Diretório de saída: outputs/results/$TIMESTAMP/"
echo ""
echo "================================================================================"
echo ""

# Criar diretório de saída
mkdir -p "outputs/results/$TIMESTAMP/losses"

# Executar cada modelo sequencialmente (mas slices em paralelo)
for model in "${MODELS[@]}"; do
    echo "--------------------------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')] Executando $model (slices em paralelo)..."
    echo "--------------------------------------------------------------------------------"
    echo ""
    
    # Usar run_parallel.sh para executar slices em paralelo
    ./scripts/run_parallel.sh "$model" "$ALL_SLICES"
    
    echo ""
    echo "[$(date '+%H:%M:%S')] ✓ $model concluído"
    echo ""
done

echo ""
echo "================================================================================"
echo "                    TODOS OS MODELOS CONCLUÍDOS"
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

echo "Estatísticas:"
echo "  - Total de modelos: ${#MODELS[@]}"
echo "  - Slices por modelo: 5 (executados em paralelo)"
echo "  - Total de experimentos: $((${#MODELS[@]} * 5))"
echo ""
echo "================================================================================"
