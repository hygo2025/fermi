# Configuração de Experimentos

## Visão Geral

Total de experimentos: 20 (4 modelos × 5 slices)

## Modelos Implementados

Ver detalhes completos das configurações em [../src/configs/README.md](../src/configs/README.md)

### GRU4Rec
- Tipo: RNN
- Paper: Hidasi et al. (2016)
- Config: src/configs/neural/gru4rec.yaml

Parâmetros:
- embedding_size: 256
- hidden_size: 256
- num_layers: 1
- dropout_prob: 0.3
- loss_type: CE
- epochs: 10
- batch_size: 4096

### NARM
- Tipo: RNN + Attention
- Paper: Li et al. (2017)
- Config: src/configs/neural/narm.yaml

Parâmetros:
- embedding_size: 256
- hidden_size: 256
- n_layers: 1
- dropout_probs: [0.25, 0.5]
- loss_type: CE
- epochs: 10
- batch_size: 4096

### STAMP
- Tipo: Attention
- Paper: Liu et al. (2018)
- Config: src/configs/neural/stamp.yaml

Parâmetros:
- embedding_size: 256
- hidden_size: 256
- loss_type: CE
- epochs: 10
- batch_size: 4096

### SASRec
- Tipo: Transformer
- Paper: Kang & McAuley (2018)
- Config: src/configs/neural/sasrec.yaml

Parâmetros:
- embedding_size: 256
- hidden_size: 256
- n_layers: 2
- n_heads: 2
- inner_size: 512
- dropout_prob: 0.5
- loss_type: CE
- epochs: 10
- batch_size: 4096

## Dados (Sliding Window)

### Estrutura

```
data/sliding_window/
├── slice_1/
│   ├── train/       (Parquet: 30 primeiros dias)
│   └── test/        (Parquet: próximos 7 dias)
├── slice_2/
│   ├── train/
│   └── test/
...
├── slice_5/
```

### Conversão para RecBole

```
recbole_data/
├── realestate_slice1/
│   ├── realestate_slice1.inter
│   ├── realestate_slice1.train.inter
│   └── realestate_slice1.test.inter
...
```

Formato .inter (TSV):
```
session_id:token    item_id:token    timestamp:float
S_1008704          150129           1709295268.972
```

## Métricas de Avaliação

RecBole built-in:
- Recall@{5, 10, 20}
- MRR@{5, 10, 20}
- NDCG@{5, 10, 20}
- Hit@{5, 10, 20}

## Matriz de Experimentos

|         | Slice 1 | Slice 2 | Slice 3 | Slice 4 | Slice 5 |
|---------|---------|---------|---------|---------|---------|
| GRU4Rec | ✓       | ✓       | ✓       | ✓       | ✓       |
| NARM    | ✓       | ✓       | ✓       | ✓       | ✓       |
| STAMP   | ✓       | ✓       | ✓       | ✓       | ✓       |
| SASRec  | ✓       | ✓       | ✓       | ✓       | ✓       |

Total: 20 experimentos

## Tempo Estimado (GPU RTX 4090)

Com configurações otimizadas (batch_size=4096):

Por modelo:
- GRU4Rec: ~15 min (5 slices)
- NARM: ~20 min (5 slices)
- STAMP: ~15 min (5 slices)
- SASRec: ~25 min (5 slices)

Total: ~1-2 horas (com GPU cooling ativo)

## Resultados

Formato de saída:
```
results/
├── raw_results.csv              (resultados individuais)
└── aggregated_results.csv       (média ± std por modelo)
```

Agregação:
- Média dos 5 slices para cada modelo
- Desvio padrão entre slices
- Seguindo metodologia de Domingues et al. (2024)
