# Referência Rápida - Fermi Benchmark

## Pipeline Completo (Ordem de Execução)

```bash
# 1. Preparar dados (sliding window)
make prepare-data
# Output: data/sliding_window/slice_{1..5}/

# 2. Converter para RecBole
make convert-recbole
# Output: recbole_data/realestate_slice{1..5}/*.inter

# 3. Executar experimentos

# Opção A: Todos os modelos
make run-all

# Opção B: Modelos individuais
make run-gru4rec    # Apenas GRU4Rec
make run-narm       # Apenas NARM
make run-stamp      # Apenas STAMP
make run-sasrec     # Apenas SASRec

# 4. Agregar resultados
make aggregate-results
# Output: results/aggregated_results.csv
```

## Execução Manual (Debugging)

### Preparar um único slice

```bash
python src/preprocessing/sliding_window_pipeline.py \
    --input /home/hygo2025/Documents/data/processed_data/enriched_events \
    --output data/sliding_window \
    --start-date 2024-03-01 \
    --n-days 6  # Apenas 1 slice
```

### Converter um único slice

```bash
python src/preprocessing/recbole_converter.py \
    --input data/sliding_window \
    --output recbole_data
```

### Executar um único modelo em um slice

```bash
python src/run_experiments.py \
    --models GRU4Rec \
    --slices 1 \
    --data-path recbole_data \
    --output-path results
```

### Executar múltiplos modelos

```bash
python src/run_experiments.py \
    --models GRU4Rec NARM STAMP \
    --slices 1 2 3 \
    --data-path recbole_data \
    --output-path results
```

## Estrutura de Dados

### Sliding Window (Parquet)

```
data/sliding_window/slice_1/
├── train/              # 5 dias de treino
│   └── *.parquet
├── test/               # 1 dia de teste
│   └── *.parquet
└── metadata.csv        # Estatísticas
```

**Colunas:** session_id, item_id, user_id, timestamp, event_type, position

### RecBole Format (.inter)

```
recbole_data/realestate_slice1/
├── realestate.inter         # Combined (train + test)
├── realestate.train.inter   # Apenas treino
└── realestate.test.inter    # Apenas teste
```

**Formato:**
```
session_id:token	item_id:token	timestamp:float
S_1008704	150129	1709295268.972
S_1008704	150129	1709295270.649
...
```

## Métricas Implementadas

### Next Item Prediction

| Métrica | Descrição | Range |
|---------|-----------|-------|
| HitRate@K | Taxa de acerto no top-K | 0-1 |
| MRR@K | Mean Reciprocal Rank | 0-1 |
| Coverage@K | % catálogo recomendado | 0-1 |
| Popularity@K | Viés para itens populares | ≥0 |

### Rest of Session

| Métrica | Descrição | Range |
|---------|-----------|-------|
| Precision@K | Itens relevantes / K | 0-1 |
| Recall@K | Itens recuperados / total relevantes | 0-1 |
| NDCG@K | Normalized DCG | 0-1 |
| MAP@K | Mean Average Precision | 0-1 |

**K values:** 5, 10, 20

## Modelos Disponíveis

| Modelo | Tipo | Parâmetros Principais |
|--------|------|----------------------|
| GRU4Rec | RNN | embedding_size=100, hidden_size=100 |
| NARM | RNN + Attention | embedding_size=100, hidden_size=100 |
| STAMP | Attention | embedding_size=100, hidden_size=100 |
| SASRec | Transformer | n_layers=2, n_heads=2, hidden_size=100 |

## Troubleshooting

### CUDA Out of Memory

```python
# Reduzir batch size em src/run_experiments.py
'train_batch_size': 256,  # Em vez de 512
'eval_batch_size': 256,
```

### Dados não encontrados

```bash
# Verificar estrutura
ls -la data/sliding_window/
ls -la recbole_data/

# Recriar se necessário
make prepare-data
make convert-recbole
```

### Erro ao importar RecBole

```bash
# Instalar/atualizar RecBole
pip install recbole --upgrade
```

## Resultados Esperados

### Format: raw_results.csv

```csv
model,slice,dataset,best_valid_score,Recall@5,Recall@10,Recall@20,MRR@10,NDCG@10,...
GRU4Rec,1,realestate_slice1,0.1234,0.0856,0.1234,0.1789,0.0567,0.0891,...
GRU4Rec,2,realestate_slice2,0.1156,0.0823,0.1156,0.1654,0.0534,0.0845,...
...
```

### Format: aggregated_results.csv

```csv
model,n_slices,Recall@10,MRR@10,NDCG@10,...
GRU4Rec,5,0.1195 ± 0.0123,0.0551 ± 0.0067,0.0868 ± 0.0089,...
NARM,5,0.1267 ± 0.0145,0.0589 ± 0.0078,0.0912 ± 0.0095,...
...
```

## Logs

```bash
# Ver logs de execução
tail -f results/logs/experiment_*.log

# Logs por slice/modelo são salvos automaticamente
```

## Comparação com Paper Original

O paper reporta (para JusBrasilRec):

| Modelo | HitRate@10 | MRR@10 |
|--------|------------|--------|
| STAN (melhor KNN) | 0.751 | 0.601 |
| NARM (melhor neural) | 0.732 | 0.579 |
| Pop (baseline) | 0.599 | 0.533 |

**Nota:** Resultados podem variar devido a diferenças de domínio (legal vs imobiliário).
