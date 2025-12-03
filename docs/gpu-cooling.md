# Sistema de Cooling GPU

## Visão Geral

Sistema automático de pausas periódicas para proteger a GPU durante treinamentos longos.

Funcionalidades:
- Pausas a cada N epochs
- Monitoramento de temperatura
- Pausa forçada se temperatura muito alta
- Logging de temperatura antes/depois

## Configuração Padrão

GPU cooling está ativado por padrão:

```bash
python src/run_experiments.py --models GRU4Rec --slices 1
```

Parâmetros default:
- Pausa a cada 5 epochs
- Duração: 60 segundos
- Temperatura máxima: 80°C

## Personalização

### Pausar mais frequentemente

```bash
python src/run_experiments.py --models GRU4Rec --slices 1 --cool-every 3
```

### Pausas mais longas

```bash
python src/run_experiments.py --models GRU4Rec --slices 1 --cool-duration 120
```

### Temperatura máxima mais baixa

```bash
python src/run_experiments.py --models GRU4Rec --slices 1 --max-temp 75
```

### Combinado (Conservador)

```bash
python src/run_experiments.py --models GRU4Rec --slices 1 \
    --cool-every 3 \
    --cool-duration 120 \
    --max-temp 75
```

### Desabilitar (Não Recomendado)

```bash
python src/run_experiments.py --models GRU4Rec --slices 1 --no-gpu-cooling
```

Usar apenas para testes curtos.

## Argumentos Disponíveis

| Argumento | Default | Descrição |
|-----------|---------|-----------|
| --enable-gpu-cooling | True | Ativa sistema de cooling |
| --no-gpu-cooling | - | Desativa sistema de cooling |
| --cool-every | 5 | Pausar a cada N epochs |
| --cool-duration | 60 | Duração da pausa (segundos) |
| --max-temp | 80 | Temperatura máxima (°C) |

## Saída Durante Execução

Exemplo de pausa programada:

```
================================================================================
GPU COOLING BREAK (Epoch 5)
Reason: scheduled
Temperature before: 76°C
Waiting 60 seconds...
================================================================================

60s remaining (GPU: 76°C)
50s remaining (GPU: 74°C)
40s remaining (GPU: 72°C)
30s remaining (GPU: 70°C)
20s remaining (GPU: 68°C)
10s remaining (GPU: 66°C)

================================================================================
COOLING COMPLETE
Temperature: 76°C → 66°C (Delta: -10°C)
Resuming training...
================================================================================
```

## Impacto no Tempo

### Sem Cooling

- 10 epochs × 30s = 5 minutos
- Temperatura final: ~82°C
- Risco: Throttling, desgaste acelerado

### Com Cooling (default)

- 10 epochs × 30s = 5 minutos
- 2 pausas × 60s = 2 minutos
- Total: 7 minutos (+40%)
- Temperatura: ~70-75°C

Tradeoff: +40% tempo = 50% menos desgaste térmico

## Temperaturas de Referência

| Temperatura | Status | Ação |
|-------------|--------|------|
| < 70°C | Excelente | Continuar |
| 70-75°C | Ótimo | Default OK |
| 75-80°C | Aceitável | Considerar --cool-every 3 |
| 80-85°C | Alto | Usar --max-temp 75 |
| > 85°C | Crítico | Parar experimento |

RTX 4090 Max Temperature: 90°C

## Recomendações por Cenário

### Teste Rápido (1 slice)

```bash
# Default OK
python src/run_experiments.py --models GRU4Rec --slices 1
```

### Múltiplos Slices Paralelos

```bash
# Mais conservador
./scripts/run_parallel.sh GRU4Rec "1 2 3"

# Com cooling agressivo
python src/run_experiments.py --models GRU4Rec --slices 1 2 3 \
    --cool-every 3 \
    --max-temp 75
```

### Experimento Longo (run-all)

```bash
# Pausas mais longas
python src/run_experiments.py --all-slices --cool-duration 90
```

### Ambiente Quente (Verão)

```bash
# Muito conservador
python src/run_experiments.py --all-slices \
    --cool-every 3 \
    --cool-duration 120 \
    --max-temp 75
```

### Ambiente Frio (Inverno)

```bash
# Pode ser menos conservador
python src/run_experiments.py --all-slices \
    --cool-every 7 \
    --max-temp 82
```

## Monitoramento

### Durante Treino

Terminal 1 - Experimento:
```bash
python src/run_experiments.py --models GRU4Rec --slices 1
```

Terminal 2 - GPU:
```bash
watch -n 1 nvidia-smi
```

### Verificar Logs

```bash
# Ver quando pausas ocorreram
grep "COOLING" results/logs/experiment_*.log

# Ver temperaturas registradas
grep "Temperature" results/logs/experiment_*.log
```

## Troubleshooting

### GPU continua esquentando

Soluções:
1. Aumentar frequência: --cool-every 2
2. Pausas mais longas: --cool-duration 120
3. Reduzir batch size nos configs
4. Verificar ventilação do PC

### Pausas muito frequentes

Soluções:
1. Menos pausas: --cool-every 7
2. Pausas mais curtas: --cool-duration 30
3. Verificar limpeza dos fans
4. Melhorar airflow do gabinete

### nvidia-smi não funciona

O sistema vai funcionar apenas com pausas fixas (--cool-every), sem monitorar temperatura.

## Implementação

Código relevante:
- src/utils/gpu_cooling.py - Implementação do callback
- src/run_experiments.py - Integração automática

Teste standalone:
```bash
python src/utils/gpu_cooling.py
```

## Benefícios

- Maior longevidade da GPU
- Evita thermal throttling
- Menos crashes por superaquecimento
- Fans mais silenciosos
- Menor consumo de energia

## Custo

- +20-40% tempo total de execução

Para experimentos longos (> 1 hora), vale a pena.
