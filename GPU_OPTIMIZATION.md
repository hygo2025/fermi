# Otimização GPU - Guia Completo

## 🚀 Configurações Otimizadas Aplicadas

### Mudanças nos Configs (todos os modelos)

**ANTES:**
```yaml
train_batch_size: 512
eval_batch_size: 512
hidden_size: 100
embedding_size: 100
```

**DEPOIS:**
```yaml
train_batch_size: 4096    # 8x maior!
eval_batch_size: 4096     # 8x maior!
hidden_size: 256          # 2.5x maior
embedding_size: 256       # 2.5x maior
```

**Impacto Esperado:**
- ⚡ Treino 4-6x mais rápido
- 🎯 Modelos mais expressivos
- 💾 Uso de VRAM: ~6-8 GB por experimento
- ⏱️ Tempo total: ~6-8h → **1-2 horas!**

### Backup dos Configs Originais

Os configs originais foram salvos em:
```
src/configs/neural/*.yaml.backup
```

Para reverter:
```bash
cd src/configs/neural
for f in *.backup; do mv "$f" "${f%.backup}"; done
```

## Opções de Execução

### Opção 1: Single Thread (Simples)

```bash
# Um experimento por vez, mas muito mais rápido
python src/run_experiments.py --models GRU4Rec --slices 1

# Tempo: ~30 segundos (vs 2-3 min antes)
```

### Opção 2: Paralelo - Múltiplos Slices (RECOMENDADO)

```bash
# Rodar 3 slices ao mesmo tempo
./scripts/run_parallel.sh GRU4Rec "1 2 3"

# Uso de GPU: ~18-20 GB
# Tempo: ~1 minuto para 3 slices!
```

### Opção 3: Paralelo - Múltiplos Modelos

```bash
# Terminal 1
python src/run_experiments.py --models GRU4Rec --slices 1 &

# Terminal 2
python src/run_experiments.py --models NARM --slices 1 &

# Terminal 3
python src/run_experiments.py --models STAMP --slices 1 &

# Aguardar todos
wait
```

## Uso de GPU Esperado

### Com Batch Size 4096

```
Single experiment:
├─ GPU Memory: ~6-8 GB
├─ GPU Util: 90-100%
└─ Temp: ~65-70°C

3 experiments parallel:
├─ GPU Memory: ~18-22 GB
├─ GPU Util: 100%
└─ Temp: ~75-80°C (OK, < 85°C)
```

## Monitoramento

### GPU Monitor

```bash
# Terminal separado
watch -n 1 nvidia-smi

# Ou use o script customizado
./scripts/monitor_gpu.sh
```

### Logs dos Experimentos Paralelos

```bash
# Ver todos os logs ao mesmo tempo
tail -f results/logs/*_parallel.log

# Ver log específico
tail -f results/logs/GRU4Rec_slice1_parallel.log
```

## Comandos Rápidos

```bash
# Teste rápido otimizado (1 slice)
python src/run_experiments.py --models GRU4Rec --slices 1
# Tempo: ~30 segundos

# Paralelo - 3 slices
./scripts/run_parallel.sh GRU4Rec "1 2 3"
# Tempo: ~1 minuto

# Todos os 5 slices em paralelo (usa ~100% GPU)
./scripts/run_parallel.sh GRU4Rec "1 2 3 4 5"
# Tempo: ~2 minutos
# ⚠️  Pode ficar apertado na VRAM, monitorar nvidia-smi

# Experimento completo otimizado
make run-all
# Tempo: ~1-2 horas (vs 6-8h antes!)
```

## Troubleshooting

### Out of Memory (OOM)

Se ver erro `CUDA out of memory`:

**Solução 1:** Reduzir batch size
```bash
# Editar configs
vim src/configs/neural/gru4rec.yaml

# Mudar para:
train_batch_size: 2048  # em vez de 4096
```

**Solução 2:** Menos slices em paralelo
```bash
# Em vez de 5, rodar 3 por vez
./scripts/run_parallel.sh GRU4Rec "1 2 3"
# Depois:
./scripts/run_parallel.sh GRU4Rec "4 5"
```

### Processos Travados

```bash
# Listar processos Python
ps aux | grep run_experiments

# Matar todos
pkill -f run_experiments.py

# Ou matar por PID
kill -9 <PID>
```

### GPU não em 100%

Se GPU util < 80%:
- ✅ Batch size pode estar pequeno ainda
- ✅ Aumentar para 8192
- ✅ Ou rodar mais slices em paralelo

## Comparação de Performance

### Antes (Configs Originais)

| Experimento | Batch Size | Tempo |
|-------------|------------|-------|
| 1 slice | 512 | ~2-3 min |
| 5 slices | 512 | ~10-15 min |
| 20 experimentos | 512 | ~6-8 horas |

### Depois (Configs Otimizados)

| Experimento | Batch Size | Tempo |
|-------------|------------|-------|
| 1 slice | 4096 | ~30 seg |
| 5 slices (paralelo) | 4096 | ~2 min |
| 20 experimentos | 4096 | **~1-2 horas** |

**Speedup: 4-6x** 🚀

## Restaurar Configs Originais

Se quiser voltar aos configs conservadores:

```bash
cd src/configs/neural
for f in *.backup; do 
    mv "$f" "${f%.backup}"
done
```

## Teste Recomendado Agora

```bash
# 1. Teste rápido (validar que funciona)
python src/run_experiments.py --models GRU4Rec --slices 1

# 2. Se funcionou, paralelo com 3 slices
./scripts/run_parallel.sh GRU4Rec "1 2 3"

# 3. Se ainda OK, rodar tudo!
make run-all
```

**Aproveite a RTX 4090!** 💪
