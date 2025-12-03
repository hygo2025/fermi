# Otimização GPU

## Configurações Aplicadas

As configurações foram otimizadas para maximizar o uso da GPU RTX 4090:

### Antes (Conservador)

```yaml
train_batch_size: 512
eval_batch_size: 512
hidden_size: 100
embedding_size: 100
```

### Depois (Otimizado)

```yaml
train_batch_size: 4096    # 8x maior
eval_batch_size: 4096     # 8x maior
hidden_size: 256          # 2.5x maior
embedding_size: 256       # 2.5x maior
```

### Impacto

- Speedup: 4-6x mais rápido
- Uso de VRAM: ~6-8 GB por experimento
- Tempo total: 6-8h → 1-2 horas
- Modelos mais expressivos

## Uso de GPU por Cenário

### Single Experiment

```bash
python src/run_experiments.py --models GRU4Rec --slices 1
```

GPU:
- Memory: ~6-8 GB / 24 GB
- Utilization: 90-100%
- Temperature: ~65-70°C
- Tempo: ~30 segundos

### 3 Experiments Parallel

```bash
./scripts/run_parallel.sh GRU4Rec "1 2 3"
```

GPU:
- Memory: ~18-22 GB / 24 GB
- Utilization: 100%
- Temperature: ~75-80°C
- Tempo: ~1 minuto

### 5 Experiments Parallel (Máximo)

```bash
./scripts/run_parallel.sh GRU4Rec "1 2 3 4 5"
```

GPU:
- Memory: ~24 GB / 24 GB (100%)
- Utilization: 100%
- Temperature: ~80°C
- Tempo: ~2 minutos

Atenção: Monitorar VRAM, pode haver out of memory.

## Paralelização

### Script Automático

```bash
# Executar N slices em background
./scripts/run_parallel.sh GRU4Rec "1 2 3"

# Ver processos
jobs

# Aguardar todos
wait

# Ver logs
tail -f results/logs/GRU4Rec_slice*_parallel.log
```

### Manual (Múltiplos Terminais)

```bash
# Terminal 1
python src/run_experiments.py --models GRU4Rec --slices 1 &

# Terminal 2
python src/run_experiments.py --models NARM --slices 1 &

# Terminal 3
python src/run_experiments.py --models STAMP --slices 1 &
```

## Ajuste de Batch Size

Se encontrar out of memory, edite os configs:

```bash
vim src/configs/neural/gru4rec.yaml
```

Opções:
- Conservador: batch_size = 2048
- Balanceado: batch_size = 4096 (default)
- Agressivo: batch_size = 8192

## Monitoramento

### nvidia-smi

```bash
watch -n 1 nvidia-smi
```

Observar:
- GPU Memory Usage (MB)
- GPU Utilization (%)
- Temperature (°C)
- Power Draw (W)

### Monitor Customizado

```bash
./scripts/monitor_gpu.sh
```

Mostra:
- Status da GPU
- Temperatura
- Utilization
- VRAM usage
- Progresso do experimento (últimas linhas do log)

## Restaurar Configurações Originais

Os configs otimizados têm backups em src/configs/neural/*.yaml.backup

Para reverter:

```bash
cd src/configs/neural
for f in *.backup; do mv "$f" "${f%.backup}"; done
```

Quando reverter:
- Debugging de erros
- GPU com pouca VRAM (< 16GB)
- Testes de comparação
- Reproduzir paper original

## Comparação de Performance

### Antes (batch_size=512)

| Experimento | Tempo |
|-------------|-------|
| 1 slice | 2-3 min |
| 5 slices | 10-15 min |
| 20 experimentos | 6-8 horas |

### Depois (batch_size=4096)

| Experimento | Tempo |
|-------------|-------|
| 1 slice | 30 seg |
| 5 slices (paralelo) | 2 min |
| 20 experimentos | 1-2 horas |

Speedup: 4-6x
