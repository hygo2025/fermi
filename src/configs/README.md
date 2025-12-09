# Configuracoes de Modelos

Arquivos YAML com configuracoes dos modelos de recomendacao baseada em sessao.

## Modelos Disponiveis

| Modelo  | Arquivo              | Tipo            | Paper                 |
|---------|----------------------|-----------------|-----------------------|
| GRU4Rec | neural/gru4rec.yaml  | RNN             | Hidasi et al. (2016 ) |
| NARM    | neural/narm.yaml     | RNN + Attention | Li et al. (2017)      |
| STAMP   | neural/stamp.yaml    | Attention       | Liu et al. (2018)     |
| SASRec  | neural/sasrec.yaml   | Transformer     | Kang & McAuley (2018) |

## Estrutura dos Arquivos YAML

```yaml
# Modelo
model: GRU4Rec

# Dataset (sobrescrito pelo runner)
dataset: realestate_slice1
data_path: recbole_data/

# Treinamento
epochs: 10
train_batch_size: 4096
eval_batch_size: 4096
learning_rate: 0.001

# Parametros do Modelo
embedding_size: 256
hidden_size: 256

# Avaliacao
metrics: ['Recall', 'MRR', 'NDCG', 'Hit']
topk: [5, 10, 20]
valid_metric: Recall@10

# Campos de Sessao
SESSION_ID_FIELD: session_id
ITEM_ID_FIELD: item_id
TIME_FIELD: timestamp
USER_ID_FIELD: session_id

# Protocolo de Avaliacao
eval_args:
  split: {'RS': [None, None, None]}  # Usar arquivos pre-divididos
  order: 'TO'  # Ordem temporal
  mode: 'full'

# Device
device: cuda
gpu_id: 0

# Reprodutibilidade
seed: 42
```

## Modificar Configuracoes

Para ajustar hiperparametros, edite o arquivo YAML correspondente:

```bash
vim src/configs/neural/gru4rec.yaml
```

Exemplo de alteracoes:
- Aumentar epochs: epochs: 20
- Reduzir batch size: train_batch_size: 2048
- Ajustar learning rate: learning_rate: 0.0001

## Parametros Principais

### Treinamento
- epochs: Numero de epocas (default: 10)
- train_batch_size: Batch size treino (default: 4096)
- eval_batch_size: Batch size avaliacao (default: 4096)
- learning_rate: Taxa de aprendizado (default: 0.001)

### Modelo
- embedding_size: Dimensao dos embeddings (default: 256)
- hidden_size: Dimensao das camadas ocultas (default: 256)
- dropout_prob: Probabilidade de dropout (varia por modelo)

### Avaliacao
- metrics: Metricas a calcular (Recall, MRR, NDCG, Hit)
- topk: Valores de K para top-K (default: [5, 10, 20])
- valid_metric: Metrica para early stopping (default: Recall@10)

### Device
- device: 'cuda' ou 'cpu' (default: cuda)
- gpu_id: ID da GPU (default: 0)

## Configuracoes Otimizadas

As configuracoes atuais estao otimizadas para GPU RTX 4090:

Antes (conservador):
- batch_size: 512
- hidden_size: 100
- embedding_size: 100

Depois (otimizado):
- batch_size: 4096 (8x maior)
- hidden_size: 256 (2.5x maior)
- embedding_size: 256 (2.5x maior)

Backups das configuracoes originais: neural/*.yaml.backup

Para reverter:
```bash
cd src/configs/neural
for f in *.backup; do mv "$f" "${f%.backup}"; done
```

## Adicionar Novo Modelo

1. Criar arquivo YAML em neural/

```yaml
model: NovoModelo
embedding_size: 256
hidden_size: 256
# ... outros parametros
```

2. Adicionar classe em run_experiments.py:

```python
from recbole.model.sequential_recommender import NovoModelo

self.available_models = {
    'GRU4Rec': GRU4Rec,
    'NARM': NARM,
    'STAMP': STAMP,
    'SASRec': SASRec,
    'NovoModelo': NovoModelo,
}
```

3. Criar comando no Makefile:

```makefile
run-novomodelo:
	python src/run_experiments.py --models NovoModelo --all-slices
```

## Notas

- Arquivos YAML sao carregados automaticamente por run_experiments.py
- dataset e data_path sao sobrescritos pelo runner para cada slice
- Todas as configs seguem o formato RecBole
- Formato RecBole: https://recbole.io/docs/user_guide/config_settings.html
