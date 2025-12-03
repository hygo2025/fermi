# Experimentos a Serem Executados

## Visão Geral

**Total de experimentos:** 20 (4 modelos × 5 slices)

## Modelos

### 1. GRU4Rec
- **Config:** `src/configs/neural/gru4rec.yaml`
- **Tipo:** RNN
- **Paper:** Hidasi et al. (2016)
- **Parâmetros:**
  - embedding_size: 100
  - hidden_size: 100
  - num_layers: 1
  - dropout_prob: 0.3
  - loss_type: CE
  - epochs: 10
  - batch_size: 512

### 2. NARM
- **Config:** `src/configs/neural/narm.yaml`
- **Tipo:** RNN + Attention
- **Paper:** Li et al. (2017)
- **Parâmetros:**
  - embedding_size: 100
  - hidden_size: 100
  - n_layers: 1
  - dropout_probs: [0.25, 0.5]
  - loss_type: CE
  - epochs: 10
  - batch_size: 512

### 3. STAMP
- **Config:** `src/configs/neural/stamp.yaml`
- **Tipo:** Attention
- **Paper:** Liu et al. (2018)
- **Parâmetros:**
  - embedding_size: 100
  - hidden_size: 100
  - loss_type: CE
  - epochs: 10
  - batch_size: 512

### 4. SASRec
- **Config:** `src/configs/neural/sasrec.yaml`
- **Tipo:** Transformer
- **Paper:** Kang & McAuley (2018)
- **Parâmetros:**
  - n_layers: 2
  - n_heads: 2
  - hidden_size: 100
  - inner_size: 256
  - hidden_dropout_prob: 0.5
  - attn_dropout_prob: 0.5
  - epochs: 10
  - batch_size: 512

## Slices (Sliding Window)

| Slice | Train Period | Test Period | Train Events | Test Events | Train Sessions | Test Sessions |
|-------|-------------|-------------|--------------|-------------|----------------|---------------|
| 1     | Mar 01-05   | Mar 06      | 763,159      | 153,674     | 43,232         | 8,763         |
| 2     | Mar 07-11   | Mar 12      | 691,302      | 164,295     | 39,251         | 9,379         |
| 3     | Mar 13-17   | Mar 18      | 748,690      | 199,885     | 42,241         | 11,358        |
| 4     | Mar 19-23   | Mar 24      | 806,480      | 136,408     | 45,721         | 7,585         |
| 5     | Mar 25-29   | Mar 30      | 737,564      | 106,016     | 41,747         | 5,948         |

## Métricas Avaliadas

### RecBole Built-in (durante treinamento)
- Recall@{5, 10, 20}
- MRR@{5, 10, 20}
- NDCG@{5, 10, 20}
- Hit@{5, 10, 20}

### Custom Metrics (pós-processamento)
- **Next Item:**
  - HitRate@K
  - MRR@K
  - Coverage@K
  - Popularity@K
  
- **Rest of Session:**
  - Precision@K
  - Recall@K
  - NDCG@K
  - MAP@K

## Matriz de Experimentos

```
                Slice 1   Slice 2   Slice 3   Slice 4   Slice 5
GRU4Rec           ✓         ✓         ✓         ✓         ✓
NARM              ✓         ✓         ✓         ✓         ✓
STAMP             ✓         ✓         ✓         ✓         ✓
SASRec            ✓         ✓         ✓         ✓         ✓
```

**Total:** 20 experimentos

## Protocolo de Avaliação

1. **Split:** Pré-definido (train/test por slice)
2. **Ordem:** Temporal (respeitando timestamps)
3. **Modo:** Full (avaliação completa)
4. **Validação:** Mesma estratégia do teste
5. **Early Stopping:** Baseado em Recall@10

## Resultados Esperados

### Por Experimento (20 linhas)
```csv
model,slice,dataset,Recall@5,Recall@10,Recall@20,MRR@5,MRR@10,MRR@20,NDCG@10,...
GRU4Rec,1,realestate_slice1,0.0856,0.1234,0.1789,0.0512,0.0567,0.0601,0.0891,...
GRU4Rec,2,realestate_slice2,0.0823,0.1156,0.1654,0.0489,0.0534,0.0568,0.0845,...
...
```

### Agregado (4 linhas)
```csv
model,n_slices,Recall@10,MRR@10,NDCG@10,...
GRU4Rec,5,0.1195 ± 0.0123,0.0551 ± 0.0067,0.0868 ± 0.0089,...
NARM,5,0.1267 ± 0.0145,0.0589 ± 0.0078,0.0912 ± 0.0095,...
STAMP,5,0.1189 ± 0.0134,0.0545 ± 0.0071,0.0861 ± 0.0092,...
SASRec,5,0.1312 ± 0.0156,0.0612 ± 0.0085,0.0945 ± 0.0102,...
```

## Tempo Estimado

**Hardware:** NVIDIA RTX 4090 (24GB VRAM)

| Modelo | Tempo/Slice | Total (5 slices) |
|--------|-------------|------------------|
| GRU4Rec | ~15 min | ~75 min |
| NARM | ~20 min | ~100 min |
| STAMP | ~15 min | ~75 min |
| SASRec | ~25 min | ~125 min |
| **TOTAL** | - | **~6-7 horas** |

**Hardware:** CPU only (não recomendado)
- Tempo estimado: **20-30 horas**

## Como Executar

### Todos os experimentos (20 total)
```bash
make run-all
```

### Modelos individuais em todos slices
```bash
make run-gru4rec    # GRU4Rec em slices 1-5
make run-narm       # NARM em slices 1-5
make run-stamp      # STAMP em slices 1-5
make run-sasrec     # SASRec em slices 1-5
```

### Modelo específico em todos slices
```bash
python src/run_experiments.py --models GRU4Rec
```

### Modelo específico em slice específico
```bash
python src/run_experiments.py --models GRU4Rec --slices 1
```

### Múltiplos modelos em múltiplos slices
```bash
python src/run_experiments.py --models GRU4Rec NARM --slices 1 2 3
```

## Comparação com Paper Original

O paper Domingues et al. (2024) reporta para JusBrasilRec (domínio legal):

| Modelo | HitRate@10 | MRR@10 |
|--------|------------|--------|
| STAN (melhor KNN) | 0.751 | 0.601 |
| NARM (melhor neural) | 0.732 | 0.579 |
| GRU4Rec | ~0.65 | ~0.52 |
| Pop (baseline) | 0.599 | 0.533 |

**Nota:** Nossos resultados podem diferir devido ao domínio diferente (imobiliário vs legal).

## Arquivos Gerados

```
results/
├── raw_results.csv              # Todos os 20 experimentos
├── aggregated_results.csv       # Médias por modelo (4 linhas)
└── logs/
    └── experiment_YYYYMMDD_HHMMSS.log
```
