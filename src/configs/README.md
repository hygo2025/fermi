# Model Configurations

Configurações YAML para os modelos de recomendação baseada em sessão.

## Modelos Implementados

### Neural (4 modelos)

| Modelo | Arquivo | Tipo | Paper |
|--------|---------|------|-------|
| GRU4Rec | `neural/gru4rec.yaml` | RNN | Hidasi et al. (2016) |
| NARM | `neural/narm.yaml` | RNN + Attention | Li et al. (2017) |
| STAMP | `neural/stamp.yaml` | Attention | Liu et al. (2018) |
| SASRec | `neural/sasrec.yaml` | Transformer | Kang & McAuley (2018) |

## Estrutura dos Arquivos

```yaml
# Model
model: GRU4Rec

# Dataset (overridden by runner)
dataset: realestate_slice1
data_path: recbole_data/

# Training
epochs: 10
train_batch_size: 512
eval_batch_size: 512
learning_rate: 0.001

# Model Parameters
embedding_size: 100
hidden_size: 100
...

# Evaluation
metrics: ['Recall', 'MRR', 'NDCG', 'Hit']
topk: [5, 10, 20]
valid_metric: Recall@10

# Session Settings
SESSION_ID_FIELD: session_id
ITEM_ID_FIELD: item_id
TIME_FIELD: timestamp
USER_ID_FIELD: session_id

# Evaluation Protocol
eval_args:
  split: {'RS': [None, None, None]}  # Use pre-split files
  order: 'TO'  # Temporal order
  mode: 'full'

# Device
device: cuda
gpu_id: 0

# Reproducibility
seed: 42
```

## Modificar Configurações

Para ajustar hiperparâmetros, edite o arquivo YAML correspondente:

```bash
# Exemplo: Aumentar epochs do GRU4Rec
vim src/configs/neural/gru4rec.yaml

# Alterar:
epochs: 20  # Em vez de 10
```

## Adicionar Novo Modelo

1. Criar arquivo YAML em `neural/` ou `knn/`
2. Seguir estrutura dos arquivos existentes
3. Adicionar classe do modelo em `run_experiments.py`:

```python
self.available_models = {
    'GRU4Rec': GRU4Rec,
    'NARM': NARM,
    'STAMP': STAMP,
    'SASRec': SASRec,
    'NovoModelo': NovoModelo,  # Adicionar aqui
}
```

## Parâmetros Comuns

### Treinamento
- `epochs`: Número de épocas (default: 10)
- `train_batch_size`: Batch size treino (default: 512)
- `eval_batch_size`: Batch size avaliação (default: 512)
- `learning_rate`: Taxa de aprendizado (default: 0.001)

### Modelo
- `embedding_size`: Dimensão dos embeddings (default: 100)
- `hidden_size`: Dimensão das camadas ocultas (default: 100)
- `dropout_prob`: Probabilidade de dropout (varia por modelo)

### Avaliação
- `metrics`: Métricas a calcular
- `topk`: Valores de K para top-K (default: [5, 10, 20])
- `valid_metric`: Métrica para early stopping

### Device
- `device`: 'cuda' ou 'cpu'
- `gpu_id`: ID da GPU (0, 1, ...)

## Notas

- Os arquivos YAML são carregados automaticamente pelo `run_experiments.py`
- `dataset` e `data_path` são sobrescritos pelo runner para cada slice
- Todas as configs seguem o formato RecBole
