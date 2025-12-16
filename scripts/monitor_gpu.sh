#!/bin/bash
# Monitor de GPU e Progresso dos Experimentos - Versão Melhorada

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Função para encontrar diretório de execução mais recente
get_latest_run_dir() {
    ls -td outputs/results/20*/ 2>/dev/null | head -1
}

# Função para criar barra de progresso
progress_bar() {
    local current=$1
    local total=$2
    local width=30
    
    # Validar inputs
    if [ -z "$current" ] || [ -z "$total" ]; then
        echo "[No data]"
        return
    fi
    
    # Converter para inteiro e tratar valores inválidos
    current=$(echo "$current" | grep -oP '^\d+' || echo "0")
    total=$(echo "$total" | grep -oP '^\d+' || echo "100")
    
    if [ "$total" -eq 0 ]; then
        echo "[No data]"
        return
    fi
    
    # Limitar current ao máximo de total
    if [ "$current" -gt "$total" ]; then
        current=$total
    fi
    
    local percentage=$((current * 100 / total))
    local filled=$((current * width / total))
    
    # Garantir que filled não exceda width
    if [ "$filled" -gt "$width" ]; then
        filled=$width
    fi
    
    local empty=$((width - filled))
    
    # Escolher cor baseado na porcentagem
    local color=$RED
    if [ $percentage -ge 75 ]; then
        color=$GREEN
    elif [ $percentage -ge 50 ]; then
        color=$YELLOW
    fi
    
    printf "["
    printf "${color}%${filled}s${NC}" | tr ' ' '#'
    printf "%${empty}s" | tr ' ' '-'
    printf "] ${BOLD}${percentage}%%${NC}"
}

# Função para extrair progresso de epoch do log
extract_epoch_progress() {
    local log_file=$1
    
    # Procurar por "Train    5:" ou similar (RecBole format)
    local epoch_line=$(tail -200 "$log_file" 2>/dev/null | grep -E "Train\s+\d+:" | tail -1)
    
    if [ -n "$epoch_line" ]; then
        # Extrair número da época (apenas dígitos após "Train")
        local current_epoch=$(echo "$epoch_line" | grep -oP 'Train\s+\K\d+' | head -1)
        # Extrair loss se disponível
        local loss=$(echo "$epoch_line" | grep -oP 'loss:\s*\K[0-9.]+' | head -1)
        
        if [ -n "$current_epoch" ] && [ "$current_epoch" -lt 1000 ]; then
            echo "$current_epoch|$loss"
            return
        fi
    fi
    
    # Fallback: procurar por "epoch: X" ou "Epoch X"
    epoch_line=$(tail -100 "$log_file" 2>/dev/null | grep -iE "(epoch:|epoch\s+\d+)" | tail -1)
    if [ -n "$epoch_line" ]; then
        current_epoch=$(echo "$epoch_line" | grep -oP '(?i)epoch[:\s]+\K\d+' | head -1)
        if [ -n "$current_epoch" ] && [ "$current_epoch" -lt 1000 ]; then
            echo "$current_epoch|"
            return
        fi
    fi
    
    echo "0|"
}

# Função para mostrar progresso dos experimentos
show_progress() {
    latest_dir=$(get_latest_run_dir)
    
    if [ -z "$latest_dir" ]; then
        echo -e "${YELLOW}Nenhum experimento em execução ainda.${NC}"
        echo ""
        return
    fi
    
    echo -e "${CYAN}${BOLD} Diretório: ${NC}${latest_dir}"
    echo ""
    
    # Contar modelos em execução
    local active_models=0
    local completed_models=0
    local failed_models=0
    
    shopt -s nullglob
    for log in "${latest_dir}"*_execution.log; do
        if [ -f "$log" ]; then
            model=$(basename "$log" _execution.log)
            
            # Verificar se há processo Python ativo para este modelo
            local is_running=false
            if pgrep -f "python.*--models $model" > /dev/null 2>&1; then
                is_running=true
                active_models=$((active_models + 1))
            fi
            
            # Extrair progresso
            local progress=$(extract_epoch_progress "$log")
            local current_epoch=$(echo "$progress" | cut -d'|' -f1)
            local loss=$(echo "$progress" | cut -d'|' -f2)
            
            # Detectar total de epochs (padrão: 100)
            local total_epochs=100
            local epochs_config=$(grep -oP "epochs.*?(\d+)" "$log" 2>/dev/null | head -1 | grep -oP '\d+')
            if [ -n "$epochs_config" ]; then
                total_epochs=$epochs_config
            fi
            
            # Status do modelo
            local status=""
            local status_color=$NC
            
            if $is_running; then
                status="${GREEN}●${NC} Rodando"
                status_color=$GREEN
            elif grep -q "Test result" "$log" 2>/dev/null; then
                status="${BLUE}✓${NC} Completo"
                status_color=$BLUE
                completed_models=$((completed_models + 1))
            elif grep -qi "error\|exception\|failed" "$log" 2>/dev/null; then
                status="${RED}✗${NC} Erro"
                status_color=$RED
                failed_models=$((failed_models + 1))
            else
                status="${YELLOW}⏸${NC} Pausado"
                status_color=$YELLOW
            fi
            
            echo -e "${BOLD}${model}${NC} ${status}"
            
            if $is_running; then
                # Mostrar progresso de época (somente se época > 0 e < 1000)
                if [ "$current_epoch" -gt 0 ] && [ "$current_epoch" -lt 1000 ]; then
                    echo -ne "  Época: ${CYAN}${current_epoch}/${total_epochs}${NC}  "
                    progress_bar $current_epoch $total_epochs
                    echo ""
                    
                    if [ -n "$loss" ]; then
                        echo -e "  Loss: ${MAGENTA}${loss}${NC}"
                    fi
                fi
                
                # Mostrar últimas 2 linhas relevantes do log
                echo -e "${CYAN}  Status:${NC}"
                tail -100 "$log" 2>/dev/null | grep -E "INFO.*Running|INFO.*Progress|Testing|Valid|Best" | tail -2 | sed 's/^/    /' | cut -c 1-100
            else
                # Mostrar última linha relevante
                tail -50 "$log" 2>/dev/null | grep -E "INFO|Test result|Error|Exception" | tail -1 | sed 's/^/  /' | cut -c 1-100
            fi
            
            echo ""
        fi
    done
    shopt -u nullglob
    
    # Resumo geral
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Resumo:${NC} ${GREEN}●${NC} Ativos: ${GREEN}${active_models}${NC} | ${BLUE}✓${NC} Completos: ${BLUE}${completed_models}${NC} | ${RED}✗${NC} Erros: ${RED}${failed_models}${NC}"
    
    # Conta quantos experimentos já foram concluídos
    if [ -f "${latest_dir}raw_results.csv" ]; then
        total_runs=$(tail -n +2 "${latest_dir}raw_results.csv" 2>/dev/null | wc -l)
        echo -e "${BOLD}Experimentos salvos:${NC} ${total_runs}"
    fi
    
    echo ""
}

# Função para mostrar GPU
show_gpu() {
    echo -e "${BOLD} Status da GPU:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Obter informações da GPU
    local gpu_info=$(nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null)
    
    if [ -z "$gpu_info" ]; then
        echo -e "${RED}nvidia-smi não disponível${NC}"
        echo ""
        return
    fi
    
    # Parse GPU info - garantir valores inteiros válidos
    local gpu_name=$(echo "$gpu_info" | cut -d',' -f1 | xargs)
    local temp=$(echo "$gpu_info" | cut -d',' -f2 | xargs | cut -d'.' -f1)
    local gpu_util=$(echo "$gpu_info" | cut -d',' -f3 | xargs | cut -d'.' -f1)
    local mem_util=$(echo "$gpu_info" | cut -d',' -f4 | xargs | cut -d'.' -f1)
    local mem_used=$(echo "$gpu_info" | cut -d',' -f5 | xargs | cut -d'.' -f1)
    local mem_total=$(echo "$gpu_info" | cut -d',' -f6 | xargs | cut -d'.' -f1)
    local power_draw=$(echo "$gpu_info" | cut -d',' -f7 | xargs | cut -d'.' -f1)
    local power_limit=$(echo "$gpu_info" | cut -d',' -f8 | xargs | cut -d'.' -f1)
    
    # Validar e limpar valores
    temp=$(echo "$temp" | grep -oP '^\d+' || echo "0")
    gpu_util=$(echo "$gpu_util" | grep -oP '^\d+' || echo "0")
    mem_used=$(echo "$mem_used" | grep -oP '^\d+' || echo "0")
    mem_total=$(echo "$mem_total" | grep -oP '^\d+' || echo "1")
    power_draw=$(echo "$power_draw" | grep -oP '^\d+' || echo "0")
    power_limit=$(echo "$power_limit" | grep -oP '^\d+' || echo "1")
    
    echo -e "${BOLD}GPU:${NC} ${gpu_name}"
    
    # Temperatura com cores
    local temp_color=$GREEN
    if [ "$temp" -ge 80 ]; then
        temp_color=$RED
    elif [ "$temp" -ge 70 ]; then
        temp_color=$YELLOW
    fi
    echo -ne "${BOLD}Temperatura:${NC} ${temp_color}${temp}°C${NC}  "
    
    # GPU Utilization (limitar a 100%)
    if [ "$gpu_util" -gt 100 ]; then
        gpu_util=100
    fi
    echo -ne "${BOLD}GPU:${NC} ${gpu_util}%  "
    
    # Memory
    local mem_pct=0
    if [ "$mem_total" -gt 0 ]; then
        mem_pct=$((mem_used * 100 / mem_total))
    fi
    if [ "$mem_pct" -gt 100 ]; then
        mem_pct=100
    fi
    
    local mem_color=$GREEN
    if [ $mem_pct -ge 90 ]; then
        mem_color=$RED
    elif [ $mem_pct -ge 75 ]; then
        mem_color=$YELLOW
    fi
    echo -e "${BOLD}VRAM:${NC} ${mem_color}${mem_used}/${mem_total} MB (${mem_pct}%)${NC}"
    
    # Power
    local power_pct=0
    if [ "$power_limit" -gt 0 ]; then
        power_pct=$((power_draw * 100 / power_limit))
    fi
    if [ "$power_pct" -gt 100 ]; then
        power_pct=100
    fi
    echo -e "${BOLD}Power:${NC} ${power_draw}W / ${power_limit}W (${power_pct}%)"
    
    # Barra de utilização
    echo -ne "${BOLD}GPU Util:${NC}  "
    progress_bar $gpu_util 100
    echo ""
    
    echo -ne "${BOLD}VRAM Util:${NC} "
    progress_bar $mem_pct 100
    echo ""
    
    echo ""
}

# Função para mostrar processos Python
show_python_processes() {
    echo -e "${BOLD} Processos Ativos:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    local python_procs=$(ps aux | grep "python.*run_experiments.py" | grep -v grep)
    
    if [ -z "$python_procs" ]; then
        echo -e "${YELLOW}Nenhum processo de experimento ativo${NC}"
        echo ""
        return
    fi
    
    echo "$python_procs" | while read line; do
        local pid=$(echo "$line" | awk '{print $2}')
        local cpu=$(echo "$line" | awk '{print $3}')
        local mem=$(echo "$line" | awk '{print $4}')
        local cmd=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i}')
        
        # Extrair modelo do comando
        local model=$(echo "$cmd" | grep -oP -- '--models\s+\K\w+')
        
        if [ -n "$model" ]; then
            echo -e "${GREEN}●${NC} ${BOLD}PID ${pid}${NC} | ${CYAN}${model}${NC} | CPU: ${cpu}% | MEM: ${mem}%"
        else
            echo -e "${GREEN}●${NC} ${BOLD}PID ${pid}${NC} | CPU: ${cpu}% | MEM: ${mem}%"
        fi
    done
    
    echo ""
}

# Loop de monitoramento
INTERVAL=3  # Intervalo de atualização em segundos

while true; do
    clear
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║${NC}                         ${BOLD}MONITOR - $(date '+%Y-%m-%d %H:%M:%S')${NC}                        ${BOLD}${CYAN}║${NC}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    show_gpu
    show_python_processes
    show_progress
    
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}⟳ Atualizando a cada ${INTERVAL}s...${NC} ${CYAN}(Ctrl+C para sair)${NC}"
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    sleep $INTERVAL
done
