# Guia de Execução

## Pipeline Completo

Ordem de execução para reproduzir os experimentos:

### 1. Preparar Dados (Sliding Window)

```bash
make prepare-data
```

Entrada: /home/hygo2025/Documents/data/processed_data/enriched_events
Saída: data/sliding_window/slice_{1..5}/
Tempo: ~15 minutos

### 2. Converter para RecBole

```bash
make convert-recbole
```

Entrada: data/sliding_window/
Saída: recbole_data/realestate_slice{1..5}/*.inter
Tempo: ~10 minutos

Importante: Os arquivos são criados com naming correto automaticamente. Não é necessário criar links simbólicos manualmente.

### 3. Executar Experimentos

#### Opção A: Todos os modelos

```bash
make run-all
```

Tempo: ~1-2 horas (GPU)

#### Opção B: Modelos individuais

```bash
make run-gru4rec    # Apenas GRU4Rec (5 slices)
make run-narm       # Apenas NARM (5 slices)
make run-stamp      # Apenas STAMP (5 slices)
make run-sasrec     # Apenas SASRec (5 slices)
```

Tempo: ~15-25 minutos cada

#### Opção C: Granular (Python direto)

```bash
# Um modelo, um slice
python src/run_experiments.py --models GRU4Rec --slices 1

# Um modelo, múltiplos slices
python src/run_experiments.py --models GRU4Rec --slices 1 2 3

# Múltiplos modelos, todos os slices
python src/run_experiments.py --models GRU4Rec NARM --all-slices
```

### 4. Agregar Resultados

```bash
make aggregate-results
```

Entrada: results/raw_results.csv
Saída: results/aggregated_results.csv
Tempo: <1 minuto

## Execução Paralela

Para maximizar uso da GPU, execute múltiplos slices em paralelo:

```bash
# 3 slices em paralelo
./scripts/run_parallel.sh GRU4Rec "1 2 3"

# Monitorar em outro terminal
watch -n 1 nvidia-smi
```

Uso de VRAM:
- 1 experimento: ~6-8 GB
- 3 experimentos: ~18-22 GB
- 5 experimentos: ~24 GB (máximo)

## Comandos Úteis

### Verificar Dados

```bash
# Listar slices disponíveis
ls data/sliding_window/

# Verificar arquivos RecBole
ls recbole_data/realestate_slice1/

# Ver estrutura de um .inter
head -5 recbole_data/realestate_slice1/realestate_slice1.inter
```

### Monitoramento

```bash
# GPU
watch -n 1 nvidia-smi

# Ou monitor customizado
./scripts/monitor_gpu.sh

# Logs
tail -f results/logs/experiment_*.log
```

### Resultados

```bash
# Ver resultados individuais
cat results/raw_results.csv

# Ver resultados agregados
cat results/aggregated_results.csv

# Contar experimentos concluídos
wc -l results/raw_results.csv
```

## Troubleshooting

### Erro: "File realestate_slice1.inter not exist"

Reconverta os dados:
```bash
rm -rf recbole_data/
make convert-recbole
```

### Out of Memory (GPU)

Reduza batch size em src/configs/neural/*.yaml:
```yaml
train_batch_size: 2048  # em vez de 4096
eval_batch_size: 2048
```

### Experimentos muito lentos

Verifique:
- GPU está sendo usada? (nvidia-smi)
- Batch size adequado? (ver configs)
- GPU cooling muito frequente? (--cool-every 7)

### Processos travados

```bash
# Listar processos
ps aux | grep run_experiments

# Matar todos
pkill -f run_experiments.py
```

## Restaurar Configurações Originais

Se precisar voltar para configs conservadores:

```bash
cd src/configs/neural
for f in *.backup; do mv "$f" "${f%.backup}"; done
```

Configs originais:
- batch_size: 512
- hidden_size: 100
- embedding_size: 100
