# GPU Cooling System - Proteção Térmica

## 🧊 Sistema Implementado

Para proteger a RTX 4090 e maximizar sua longevidade, implementamos um sistema de cooling intervals automático.

### Como Funciona

1. **Pausas Periódicas:** A cada N epochs (default: 5), o treino pausa por 60 segundos
2. **Monitoramento de Temperatura:** Se GPU atingir temperatura máxima (default: 80°C), força pausa imediata
3. **Countdown Visual:** Mostra temperatura em tempo real durante a pausa
4. **Logging Completo:** Registra temperatura antes/depois e delta

### Exemplo de Saída

```
================================================================================
🧊 GPU COOLING BREAK (Epoch 5)
Reason: scheduled
Temperature before: 76°C
Waiting 60 seconds...
================================================================================

⏳ 60s remaining (GPU: 76°C)
⏳ 50s remaining (GPU: 74°C)
⏳ 40s remaining (GPU: 72°C)
⏳ 30s remaining (GPU: 70°C)
⏳ 20s remaining (GPU: 68°C)
⏳ 10s remaining (GPU: 66°C)

================================================================================
✅ COOLING COMPLETE
Temperature: 76°C → 66°C (Δ-10°C)
Resuming training...
================================================================================
```

## Uso

### Padrão (Ativado Automaticamente)

```bash
# GPU cooling ATIVADO por padrão
python src/run_experiments.py --models GRU4Rec --slices 1

# Pausa a cada 5 epochs por 60 segundos
# Max temp: 80°C
```

### Customizar Intervalos

```bash
# Pausar a cada 3 epochs
python src/run_experiments.py \
    --models GRU4Rec --slices 1 \
    --cool-every 3

# Pausar por 2 minutos
python src/run_experiments.py \
    --models GRU4Rec --slices 1 \
    --cool-duration 120

# Temperatura máxima mais conservadora (75°C)
python src/run_experiments.py \
    --models GRU4Rec --slices 1 \
    --max-temp 75
```

### Combinando Opções

```bash
# Muito conservador (pausa frequente + temp baixa)
python src/run_experiments.py \
    --models GRU4Rec --slices 1 \
    --cool-every 3 \
    --cool-duration 90 \
    --max-temp 75

# Agressivo (menos pausas, temp mais alta)
python src/run_experiments.py \
    --models GRU4Rec --slices 1 \
    --cool-every 10 \
    --cool-duration 30 \
    --max-temp 85
```

### Desabilitar (NÃO RECOMENDADO)

```bash
# Apenas para debug, não para treinos longos
python src/run_experiments.py \
    --models GRU4Rec --slices 1 \
    --no-gpu-cooling
```

## Argumentos Disponíveis

| Argumento | Default | Descrição |
|-----------|---------|-----------|
| `--enable-gpu-cooling` | `True` | Ativa sistema de cooling |
| `--no-gpu-cooling` | - | Desativa sistema de cooling |
| `--cool-every` | `5` | Pausar a cada N epochs |
| `--cool-duration` | `60` | Duração da pausa (segundos) |
| `--max-temp` | `80` | Temperatura máxima (°C) |

## Impacto no Tempo

### Sem Cooling

```
10 epochs × 30s = 5 minutos
Temperatura final: ~82°C
Risco: Throttling, desgaste acelerado
```

### Com Cooling (default)

```
10 epochs × 30s = 5 minutos
+ 2 pausas × 60s = 2 minutos
Total: 7 minutos (+40%)
Temperatura: mantida ~70-75°C
Benefício: Sem throttling, GPU mais saudável
```

**Vale a pena:** +40% tempo para ~50% menos desgaste térmico

## Recomendações por Cenário

### Teste Rápido (1 slice)

```bash
# Default está OK
python src/run_experiments.py --models GRU4Rec --slices 1
```

### Múltiplos Slices Paralelos

```bash
# Mais conservador (temperatura sobe mais com paralelo)
./scripts/run_parallel.sh GRU4Rec "1 2 3" \
    --cool-every 3 \
    --max-temp 75
```

### Experimento Longo (run-all)

```bash
# Default está OK, mas pode usar cooling mais longo
python src/run_experiments.py --all-slices \
    --cool-duration 90
```

### Verão / Ambiente Quente

```bash
# Muito conservador
python src/run_experiments.py --all-slices \
    --cool-every 3 \
    --cool-duration 120 \
    --max-temp 75
```

### Inverno / Ambiente Frio

```bash
# Pode ser um pouco mais agressivo
python src/run_experiments.py --all-slices \
    --cool-every 7 \
    --max-temp 82
```

## Monitoramento

### Durante Treino

```bash
# Terminal 1: Rodar experimento
python src/run_experiments.py --models GRU4Rec --slices 1

# Terminal 2: Monitorar GPU
watch -n 1 nvidia-smi
```

### Verificar Logs

```bash
# Ver quando pausas ocorreram
grep "COOLING" results/logs/experiment_*.log

# Ver temperaturas
grep "Temperature" results/logs/experiment_*.log
```

## Temperaturas Seguras

| Temperatura | Status | Ação |
|-------------|--------|------|
| < 70°C | ✅ Excelente | Continuar |
| 70-75°C | ✅ Ótimo | Default OK |
| 75-80°C | ⚠️ Aceitável | Considerar cooling mais frequente |
| 80-85°C | ⚠️ Alto | Usar --max-temp 78 |
| > 85°C | ❌ Muito alto | PARAR! Cooling agressivo |

**RTX 4090 Max Temp:** 90°C (nunca deixar chegar perto!)

## Troubleshooting

### GPU continua esquentando mesmo com cooling

```bash
# Aumentar frequência de pausas
--cool-every 2

# Aumentar duração
--cool-duration 120

# Reduzir batch size (configs)
vim src/configs/neural/gru4rec.yaml
# train_batch_size: 2048  (em vez de 4096)
```

### Pausas muito frequentes / treino muito lento

```bash
# Menos pausas
--cool-every 7

# Pausas mais curtas
--cool-duration 30

# Verificar ventilação do PC
# Limpar poeira dos fans
```

### nvidia-smi não funciona

```bash
# Cooling vai funcionar, mas sem monitorar temp
# Vai pausar apenas no intervalo fixo (--cool-every)
```

## Código Relevante

- **Implementação:** `src/utils/gpu_cooling.py`
- **Integração:** `src/run_experiments.py` (linha ~170)
- **Testes:** `python src/utils/gpu_cooling.py`

## Benefícios

✅ **Longevidade:** GPU dura mais anos  
✅ **Performance:** Evita thermal throttling  
✅ **Estabilidade:** Menos crashes por superaquecimento  
✅ **Silêncio:** Fans não ficam 100% o tempo todo  
✅ **Contas de luz:** Menos energia desperdiçada em calor  

## Custos

⚠️ **Tempo:** +20-40% tempo total  
⚠️ **Atenção:** Precisa monitorar ocasionalmente  

**Conclusão:** Vale MUITO a pena para experimentos longos!
