#!/bin/bash
# Executar múltiplos experimentos em paralelo na GPU

MODEL=${1:-GRU4Rec}
SLICES=${2:-"1 2 3"}

echo "================================================================================"
echo "                  EXECUÇÃO PARALELA - GPU RTX 4090"
echo "================================================================================"
echo ""
echo "Modelo: $MODEL"
echo "Slices: $SLICES"
echo ""
echo "Cada slice rodará em background usando a mesma GPU."
echo "Com batch_size=4096, espera-se ~6-8 GB por slice."
echo ""
echo "================================================================================"
echo ""

# Contador de processos
count=0

# Iniciar cada slice em background
for slice in $SLICES; do
    ((count++))
    echo "[$count] Iniciando $MODEL no slice $slice (background)..."
    
    python src/run_experiments.py \
        --models $MODEL \
        --slices $slice \
        > "results/logs/${MODEL}_slice${slice}_parallel.log" 2>&1 &
    
    pid=$!
    echo "    PID: $pid"
    echo "    Log: results/logs/${MODEL}_slice${slice}_parallel.log"
    echo ""
    
    # Pequeno delay para evitar race condition no RecBole
    sleep 5
done

echo "================================================================================"
echo "                        PROCESSOS EM EXECUÇÃO"
echo "================================================================================"
echo ""
echo "$count processos iniciados em background."
echo ""
echo "Monitorar progresso:"
echo "  - GPU: watch -n 1 nvidia-smi"
echo "  - Logs: tail -f results/logs/${MODEL}_slice*_parallel.log"
echo ""
echo "Aguardar conclusão:"
echo "  - wait (aguarda todos os background jobs)"
echo "  - jobs (lista processos ativos)"
echo ""
echo "Para matar todos:"
echo "  - pkill -f run_experiments.py"
echo ""
echo "================================================================================"
