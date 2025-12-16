#!/bin/bash
# Executar experimentos usando run_parallel.sh
# Uso: ./run_all_experiments.sh [MODEL1 MODEL2 ...]
#      Se nenhum modelo for especificado, executa todos

set -e

# Se argumentos forem passados, usa eles; senão usa todos os modelos
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("GRU4Rec" "NARM" "STAMP" "SASRec" "FPMC" "FOSSIL" "Random" "POP" "RPOP" "SPOP")
fi

ALL_SLICES="1 2 3 4 5"

# Gera timestamp único compartilhado
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export SHARED_TIMESTAMP="$TIMESTAMP"

echo "================================================================================"
echo "                  EXECUTAR EXPERIMENTOS"
echo "================================================================================"
echo ""
echo "Timestamp compartilhado: $TIMESTAMP"
echo "Modelos: ${MODELS[@]}"
echo "Total de modelos: ${#MODELS[@]}"
echo "Slices por modelo: $ALL_SLICES (executados em paralelo, 3 por vez)"
echo "Diretório de saída: outputs/results/$TIMESTAMP/"
echo ""
echo "================================================================================"
echo ""

# Criar diretório de saída
mkdir -p "outputs/results/$TIMESTAMP/losses"

# Executar cada modelo sequencialmente (mas slices em paralelo)
for model in "${MODELS[@]}"; do
    echo "--------------------------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')] Executando $model (slices em paralelo, lotes de 3)..."
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
