# Primeiro Teste - Guia Completo

## ✅ Hardware Verificado

- **GPU:** NVIDIA GeForce RTX 4090
- **VRAM:** 24 GB
- **CUDA:** 12.8
- **PyTorch:** 2.8.0 com GPU

**Status:** Pronto para usar! 🚀

## Teste Recomendado (5-10 minutos)

Vamos executar **GRU4Rec no slice 1** primeiro.

### Terminal 1 - Executar o experimento

```bash
cd /home/hygo2025/Development/projects/fermi
python src/run_experiments.py --models GRU4Rec --slices 1
```

### Terminal 2 (Opcional) - Monitorar GPU

```bash
cd /home/hygo2025/Development/projects/fermi
./scripts/monitor_gpu.sh
```

Ou simplesmente:
```bash
watch -n 1 nvidia-smi
```

## O que você verá

### 1. Inicialização (~30s)
```
[INFO] Starting Session-Based Recommendation Experiments
[INFO] Models: ['GRU4Rec']
[INFO] Slices: [1]
[INFO] Running GRU4Rec on realestate_slice1...
```

### 2. Carregamento de Dados (~20s)
```
[INFO] Loading dataset...
realestate_slice1 dataset created with:
  Train: 763,159 interactions
  Test: 153,674 interactions
```

### 3. Treinamento (~2 min)
```
[INFO] Training GRU4Rec...
epoch 0 training [time: 11.51s, train loss: 8766.91]
epoch 1 training [time: 10.66s, train loss: 2854.64]
epoch 2 training [time: 10.96s, train loss: 1901.33]
...
epoch 9 training [time: 10.90s, train loss: 1101.75]
```

**O loss deve DIMINUIR** - se diminui, o modelo está aprendendo! ✅

### 4. Teste (~1 min)

⚠️ **Problema Conhecido:** Pode falhar com erro "weights_only" (PyTorch 2.8)

**Se funcionar:**
```
[INFO] Testing GRU4Rec...
test result: {
  'recall@5': 0.0234,
  'recall@10': 0.0456,
  'recall@20': 0.0789,
  'mrr@10': 0.0312,
  'ndcg@10': 0.0423
}
```

**Se falhar:**
```
[ERROR] Weights only load failed...
```
→ **Isso é OK!** O treino funcionou, só o teste que tem bug do PyTorch.

## Uso de GPU

Durante o treinamento você deve ver:

```
GPU Memory Usage: ~2-4 GB / 24 GB
GPU Utilization: 80-100%
Temperature: ~50-60°C
```

## Resultados

Se o experimento completar com sucesso:

```bash
# Ver resultados
cat results/raw_results.csv

# Ver logs
ls results/logs/
```

## Próximos Passos

### Se funcionou:
```bash
# Testar com todos os slices (30-40 min)
make run-gru4rec

# Ou rodar todos os modelos (6-8 horas)
make run-all
```

### Se teve erro no teste (PyTorch 2.8):

**Opção A:** Ignorar por enquanto (treino funciona)
**Opção B:** Downgrade PyTorch:
```bash
pip install torch==2.0.1
```

## Troubleshooting

### GPU não sendo usada

Verifique:
```bash
# PyTorch vê a GPU?
python -c "import torch; print(torch.cuda.is_available())"

# Drivers corretos?
nvidia-smi
```

### Erro de memória GPU

Reduza o batch size em `src/configs/neural/gru4rec.yaml`:
```yaml
train_batch_size: 256  # em vez de 512
eval_batch_size: 256
```

### Processo muito lento

- Verifique se está usando GPU (monitor GPU util > 80%)
- Outros processos usando GPU? (nvidia-smi)

## Comandos Rápidos

```bash
# Teste rápido (1 modelo, 1 slice)
python src/run_experiments.py --models GRU4Rec --slices 1

# Teste médio (1 modelo, 5 slices)
make run-gru4rec

# Teste completo (4 modelos, 5 slices)
make run-all

# Monitorar GPU
./scripts/monitor_gpu.sh
```

## Tempo Esperado por Teste

| Teste | Modelos | Slices | Tempo (RTX 4090) |
|-------|---------|--------|------------------|
| Rápido | 1 | 1 | 5-10 min |
| Médio | 1 | 5 | 30-40 min |
| GRU4Rec + NARM | 2 | 5 | 60-90 min |
| Completo | 4 | 5 | 6-8 horas |

**Recomendação:** Comece com o teste rápido!
