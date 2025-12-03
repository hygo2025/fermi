#!/bin/bash
# Monitor de GPU e Progresso do Experimento

clear
echo "================================================================================"
echo "                    MONITOR DE EXPERIMENTOS - GPU RTX 4090"
echo "================================================================================"
echo ""

# Função para mostrar última linha do log
show_progress() {
    latest_log=$(ls -t results/logs/experiment_*.log 2>/dev/null | head -1)
    if [ -f "$latest_log" ]; then
        echo "Progresso do Experimento:"
        echo "-------------------------"
        tail -5 "$latest_log" | grep -E "epoch|INFO|Training|Testing" || echo "Aguardando início..."
        echo ""
    else
        echo "Nenhum experimento em execução ainda."
        echo ""
    fi
}

# Função para mostrar GPU
show_gpu() {
    echo "Status da GPU:"
    echo "--------------"
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU: %s\nTemp: %s°C | Util: %s%% | VRAM: %s/%s MB\n", $1, $2, $3, $5, $6}'
    echo ""
}

# Loop de monitoramento
while true; do
    clear
    echo "================================================================================"
    echo "                    MONITOR DE EXPERIMENTOS - GPU RTX 4090"
    echo "                          $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"
    echo ""
    
    show_gpu
    echo ""
    show_progress
    
    echo "================================================================================"
    echo "Atualizando em 2 segundos... (Ctrl+C para sair)"
    echo "================================================================================"
    
    sleep 2
done
