# Experiment Results - 2025-12-16 11:30:53

## Execution Summary

- **Timestamp:** 2025-12-16 11:30:53
- **Total Models:** 8
- **Total Slices:** 5
- **Total Experiments:** 39 (8 models × 5 slices)
- **Metrics Evaluated:** 12 (Recall, MRR, NDCG, Hit @ K=5,10,20)

## Best Models by Metric

- **Recall@10:** SASRec (`0.9388±0.0016`)
- **MRR@10:** SASRec (`0.9133±0.0021`)
- **NDCG@10:** SASRec (`0.9195±0.0020`)
- **Hit@10:** SASRec (`0.9388±0.0016`)


## Models Evaluated

### [Neural] GRU4Rec
- Experiments: 5
- Slices: [1, 2, 3, 4, 5]
- **Recall@10:** 0.9232±0.0024
- **MRR@10:** 0.8995±0.0030
- **NDCG@10:** 0.9053±0.0028
- **Hit@10:** 0.9232±0.0024

### [Neural] NARM
- Experiments: 5
- Slices: [1, 2, 3, 4, 5]
- **Recall@10:** 0.9347±0.0017
- **MRR@10:** 0.9096±0.0024
- **NDCG@10:** 0.9157±0.0022
- **Hit@10:** 0.9347±0.0017

### [Baseline] POP
- Experiments: 5
- Slices: [1, 2, 3, 4, 5]
- **Recall@10:** 0.0243±0.0020
- **MRR@10:** 0.0082±0.0011
- **NDCG@10:** 0.0118±0.0013
- **Hit@10:** 0.0243±0.0020

### [Baseline] RPOP
- Experiments: 5
- Slices: [1, 2, 3, 4, 5]
- **Recall@10:** 0.0350±0.0016
- **MRR@10:** 0.0126±0.0009
- **NDCG@10:** 0.0178±0.0011
- **Hit@10:** 0.0350±0.0016

### [Baseline] Random
- Experiments: 5
- Slices: [1, 2, 3, 4, 5]
- **Recall@10:** 0.0004±0.0001
- **MRR@10:** 0.0001±0.0000
- **NDCG@10:** 0.0002±0.0001
- **Hit@10:** 0.0004±0.0001

### [Neural] SASRec
- Experiments: 4
- Slices: [1, 2, 4, 5]
- **Recall@10:** 0.9388±0.0016
- **MRR@10:** 0.9133±0.0021
- **NDCG@10:** 0.9195±0.0020
- **Hit@10:** 0.9388±0.0016

### [Baseline] SPOP
- Experiments: 5
- Slices: [1, 2, 3, 4, 5]
- **Recall@10:** 0.9220±0.0019
- **MRR@10:** 0.7209±0.0012
- **NDCG@10:** 0.7707±0.0013
- **Hit@10:** 0.9220±0.0019

### [Neural] STAMP
- Experiments: 5
- Slices: [1, 2, 3, 4, 5]
- **Recall@10:** 0.9028±0.0037
- **MRR@10:** 0.8741±0.0089
- **NDCG@10:** 0.8811±0.0075
- **Hit@10:** 0.9028±0.0037

## Temporal Slices

### Slice 1
- Models evaluated: 8
- Best model: **SASRec** (Recall@10: 0.9387)
- Mean Recall@10: 0.5864 ± 0.4679
- Mean MRR@10: 0.5444 ± 0.4487

### Slice 2
- Models evaluated: 8
- Best model: **SASRec** (Recall@10: 0.9366)
- Mean Recall@10: 0.5836 ± 0.4666
- Mean MRR@10: 0.5407 ± 0.4458

### Slice 3
- Models evaluated: 7
- Best model: **NARM** (Recall@10: 0.9352)
- Mean Recall@10: 0.5352 ± 0.4825
- Mean MRR@10: 0.4904 ± 0.4567

### Slice 4
- Models evaluated: 8
- Best model: **SASRec** (Recall@10: 0.9401)
- Mean Recall@10: 0.5857 ± 0.4691
- Mean MRR@10: 0.5421 ± 0.4474

### Slice 5
- Models evaluated: 8
- Best model: **SASRec** (Recall@10: 0.9397)
- Mean Recall@10: 0.5844 ± 0.4689
- Mean MRR@10: 0.5410 ± 0.4473

## Top 3 Models (by Recall@10)

### 1. SASRec

| Metric | @5 | @10 | @20 |
|--------|-----|-----|-----|
| **Recall** | 0.9305±0.0017 | 0.9388±0.0016 | 0.9456±0.0019 |
| **MRR** | 0.9122±0.0022 | 0.9133±0.0021 | 0.9138±0.0022 |
| **NDCG** | 0.9169±0.0020 | 0.9195±0.0020 | 0.9212±0.0020 |
| **Hit** | 0.9305±0.0017 | 0.9388±0.0016 | 0.9456±0.0019 |

### 2. NARM

| Metric | @5 | @10 | @20 |
|--------|-----|-----|-----|
| **Recall** | 0.9273±0.0015 | 0.9347±0.0017 | 0.9416±0.0016 |
| **MRR** | 0.9086±0.0024 | 0.9096±0.0024 | 0.9100±0.0024 |
| **NDCG** | 0.9133±0.0021 | 0.9157±0.0022 | 0.9175±0.0022 |
| **Hit** | 0.9273±0.0015 | 0.9347±0.0017 | 0.9416±0.0016 |

### 3. GRU4Rec

| Metric | @5 | @10 | @20 |
|--------|-----|-----|-----|
| **Recall** | 0.9159±0.0023 | 0.9232±0.0024 | 0.9296±0.0019 |
| **MRR** | 0.8985±0.0030 | 0.8995±0.0030 | 0.8999±0.0030 |
| **NDCG** | 0.9029±0.0029 | 0.9053±0.0028 | 0.9069±0.0028 |
| **Hit** | 0.9159±0.0023 | 0.9232±0.0024 | 0.9296±0.0019 |

## Generated Files

- `raw_results.csv` - Complete results for all 39 experiments
- `aggregated_results.csv` - Aggregated metrics (mean ± std across slices)
- `results_table.md` - Markdown table with key metrics
- `metrics_comparison.png` - Bar chart comparing models (with error bars)
- `performance_heatmap.png` - Normalized performance heatmap
- `slice_consistency.png` - Performance consistency across temporal slices

## Methodology

### Data Split
- **Temporal Slices:** 5 slices using sliding window protocol
- **Protocol:** Next-item prediction
- **Evaluation:** Leave-one-out on test set

### Metrics Aggregation
- **Mean:** Average performance across 5 temporal slices
- **Std:** Sample standard deviation (ddof=1)
- **Format:** `mean±std`

### Metrics Explanation
- **Recall@K:** Proportion of relevant items in top-K recommendations
- **MRR@K:** Mean Reciprocal Rank at K (position of first relevant item)
- **NDCG@K:** Normalized Discounted Cumulative Gain at K (ranking quality)
- **Hit@K:** Whether at least one relevant item appears in top-K

Higher values are better for all metrics.

## Performance Summary

### Overall Statistics
- **Recall@10:** 0.5761 ± 0.4457 (range: 0.0002 - 0.9401)
- **MRR@10:** 0.5328 ± 0.4252 (range: 0.0001 - 0.9152)
- **NDCG@10:** 0.5434 ± 0.4296 (range: 0.0001 - 0.9212)
- **Hit@10:** 0.5761 ± 0.4457 (range: 0.0002 - 0.9401)


### Model Type Comparison

**Neural Models (4 models):**
- Recall@10: 0.9242
- MRR@10: 0.8984
- NDCG@10: 0.9047

**Baseline Models (4 models):**
- Recall@10: 0.2454
- MRR@10: 0.1855
- NDCG@10: 0.2001


## Statistical Analysis

### Variance Analysis

**Recall@10:**
- Between-model variance: 0.219305
- Within-model variance: 0.000004
- Ratio: 50980.14x

**MRR@10:**
- Between-model variance: 0.200280
- Within-model variance: 0.000013
- Ratio: 15667.15x


---

**Generated:** 2025-12-16 11:30:53  
**Total Duration:** See experiment logs for detailed timing
