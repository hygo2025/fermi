# Experiment Results - 2025-12-11 10:12:32

## Execution Summary

- **Timestamp:** 2025-12-11 10:12:32
- **Total Models:** 3
- **Total Slices:** 5
- **Total Experiments:** 11 (3 models × 5 slices)
- **Metrics Evaluated:** 12 (Recall, MRR, NDCG, Hit @ K=5,10,20)

## Best Models by Metric

- **Recall@10:** SPOP (`0.9229±nan`)
- **MRR@10:** SPOP (`0.7223±nan`)
- **NDCG@10:** SPOP (`0.7720±nan`)
- **Hit@10:** SPOP (`0.9229±nan`)


## Models Evaluated

### [Baseline] POP
- Experiments: 5
- Slices: [1, 2, 3, 4, 5]
- **Recall@10:** 0.0243±0.0020
- **MRR@10:** 0.0082±0.0011
- **NDCG@10:** 0.0118±0.0013
- **Hit@10:** 0.0243±0.0020

### [Baseline] Random
- Experiments: 5
- Slices: [1, 2, 3, 4, 5]
- **Recall@10:** 0.0004±0.0001
- **MRR@10:** 0.0001±0.0000
- **NDCG@10:** 0.0002±0.0001
- **Hit@10:** 0.0004±0.0001

### [Baseline] SPOP
- Experiments: 1
- Slices: [1]
- **Recall@10:** 0.9229±nan
- **MRR@10:** 0.7223±nan
- **NDCG@10:** 0.7720±nan
- **Hit@10:** 0.9229±nan

## Temporal Slices

### Slice 1
- Models evaluated: 3
- Best model: **SPOP** (Recall@10: 0.9229)
- Mean Recall@10: 0.3168 ± 0.5251
- Mean MRR@10: 0.2439 ± 0.4143

### Slice 2
- Models evaluated: 2
- Best model: **POP** (Recall@10: 0.0255)
- Mean Recall@10: 0.0130 ± 0.0177
- Mean MRR@10: 0.0045 ± 0.0061

### Slice 3
- Models evaluated: 2
- Best model: **POP** (Recall@10: 0.0233)
- Mean Recall@10: 0.0119 ± 0.0162
- Mean MRR@10: 0.0040 ± 0.0055

### Slice 4
- Models evaluated: 2
- Best model: **POP** (Recall@10: 0.0236)
- Mean Recall@10: 0.0119 ± 0.0165
- Mean MRR@10: 0.0041 ± 0.0057

### Slice 5
- Models evaluated: 2
- Best model: **POP** (Recall@10: 0.0220)
- Mean Recall@10: 0.0112 ± 0.0153
- Mean MRR@10: 0.0034 ± 0.0046

## Top 3 Models (by Recall@10)

### 1. SPOP

| Metric | @5 | @10 | @20 |
|--------|-----|-----|-----|
| **Recall** | 0.8861±nan | 0.9229±nan | 0.9293±nan |
| **MRR** | 0.7171±nan | 0.7223±nan | 0.7228±nan |
| **NDCG** | 0.7597±nan | 0.7720±nan | 0.7736±nan |
| **Hit** | 0.8861±nan | 0.9229±nan | 0.9293±nan |

### 2. POP

| Metric | @5 | @10 | @20 |
|--------|-----|-----|-----|
| **Recall** | 0.0135±0.0014 | 0.0243±0.0020 | 0.0405±0.0042 |
| **MRR** | 0.0068±0.0010 | 0.0082±0.0011 | 0.0092±0.0012 |
| **NDCG** | 0.0084±0.0011 | 0.0118±0.0013 | 0.0159±0.0018 |
| **Hit** | 0.0135±0.0014 | 0.0243±0.0020 | 0.0405±0.0042 |

### 3. Random

| Metric | @5 | @10 | @20 |
|--------|-----|-----|-----|
| **Recall** | 0.0002±0.0001 | 0.0004±0.0001 | 0.0008±0.0002 |
| **MRR** | 0.0001±0.0001 | 0.0001±0.0000 | 0.0001±0.0001 |
| **NDCG** | 0.0001±0.0000 | 0.0002±0.0001 | 0.0003±0.0001 |
| **Hit** | 0.0002±0.0001 | 0.0004±0.0001 | 0.0008±0.0002 |

## Generated Files

- `raw_results.csv` - Complete results for all 11 experiments
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
- **Recall@10:** 0.0951 ± 0.2748 (range: 0.0002 - 0.9229)
- **MRR@10:** 0.0694 ± 0.2166 (range: 0.0001 - 0.7223)
- **NDCG@10:** 0.0756 ± 0.2310 (range: 0.0001 - 0.7720)
- **Hit@10:** 0.0951 ± 0.2748 (range: 0.0002 - 0.9229)


### Model Type Comparison

**Neural Models (0 models):**
- Recall@10: nan
- MRR@10: nan
- NDCG@10: nan

**Baseline Models (3 models):**
- Recall@10: 0.0951
- MRR@10: 0.0694
- NDCG@10: 0.0756


## Statistical Analysis

### Variance Analysis

**Recall@10:**
- Between-model variance: 0.276528
- Within-model variance: 0.000002
- Ratio: 142173.96x

**MRR@10:**
- Between-model variance: 0.171934
- Within-model variance: 0.000001
- Ratio: 308401.95x


---

**Generated:** 2025-12-11 10:12:32  
**Total Duration:** See experiment logs for detailed timing
