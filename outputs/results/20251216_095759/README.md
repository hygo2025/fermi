# Experiment Results - 2025-12-16 10:11:43

## Execution Summary

- **Timestamp:** 2025-12-16 10:11:43
- **Total Models:** 3
- **Total Slices:** 5
- **Total Experiments:** 15 (3 models × 5 slices)
- **Metrics Evaluated:** 12 (Recall, MRR, NDCG, Hit @ K=5,10,20)

## Best Models by Metric

- **Recall@10:** NARM (`0.9347±0.0017`)
- **MRR@10:** NARM (`0.9096±0.0024`)
- **NDCG@10:** NARM (`0.9157±0.0022`)
- **Hit@10:** NARM (`0.9347±0.0017`)


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

### [Neural] STAMP
- Experiments: 5
- Slices: [1, 2, 3, 4, 5]
- **Recall@10:** 0.9028±0.0037
- **MRR@10:** 0.8741±0.0089
- **NDCG@10:** 0.8811±0.0075
- **Hit@10:** 0.9028±0.0037

## Temporal Slices

### Slice 1
- Models evaluated: 3
- Best model: **NARM** (Recall@10: 0.9351)
- Mean Recall@10: 0.9217 ± 0.0146
- Mean MRR@10: 0.8986 ± 0.0123

### Slice 2
- Models evaluated: 3
- Best model: **NARM** (Recall@10: 0.9318)
- Mean Recall@10: 0.9173 ± 0.0158
- Mean MRR@10: 0.8914 ± 0.0165

### Slice 3
- Models evaluated: 3
- Best model: **NARM** (Recall@10: 0.9352)
- Mean Recall@10: 0.9221 ± 0.0140
- Mean MRR@10: 0.8972 ± 0.0153

### Slice 4
- Models evaluated: 3
- Best model: **NARM** (Recall@10: 0.9360)
- Mean Recall@10: 0.9211 ± 0.0180
- Mean MRR@10: 0.8932 ± 0.0256

### Slice 5
- Models evaluated: 3
- Best model: **NARM** (Recall@10: 0.9355)
- Mean Recall@10: 0.9190 ± 0.0184
- Mean MRR@10: 0.8916 ± 0.0220

## Top 3 Models (by Recall@10)

### 1. NARM

| Metric | @5 | @10 | @20 |
|--------|-----|-----|-----|
| **Recall** | 0.9273±0.0015 | 0.9347±0.0017 | 0.9416±0.0016 |
| **MRR** | 0.9086±0.0024 | 0.9096±0.0024 | 0.9100±0.0024 |
| **NDCG** | 0.9133±0.0021 | 0.9157±0.0022 | 0.9175±0.0022 |
| **Hit** | 0.9273±0.0015 | 0.9347±0.0017 | 0.9416±0.0016 |

### 2. GRU4Rec

| Metric | @5 | @10 | @20 |
|--------|-----|-----|-----|
| **Recall** | 0.9159±0.0023 | 0.9232±0.0024 | 0.9296±0.0019 |
| **MRR** | 0.8985±0.0030 | 0.8995±0.0030 | 0.8999±0.0030 |
| **NDCG** | 0.9029±0.0029 | 0.9053±0.0028 | 0.9069±0.0028 |
| **Hit** | 0.9159±0.0023 | 0.9232±0.0024 | 0.9296±0.0019 |

### 3. STAMP

| Metric | @5 | @10 | @20 |
|--------|-----|-----|-----|
| **Recall** | 0.8939±0.0048 | 0.9028±0.0037 | 0.9111±0.0033 |
| **MRR** | 0.8729±0.0092 | 0.8741±0.0089 | 0.8747±0.0089 |
| **NDCG** | 0.8782±0.0080 | 0.8811±0.0075 | 0.8832±0.0074 |
| **Hit** | 0.8939±0.0048 | 0.9028±0.0037 | 0.9111±0.0033 |

## Generated Files

- `raw_results.csv` - Complete results for all 15 experiments
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
- **Recall@10:** 0.9203 ± 0.0139 (range: 0.8991 - 0.9360)
- **MRR@10:** 0.8944 ± 0.0163 (range: 0.8642 - 0.9125)
- **NDCG@10:** 0.9007 ± 0.0157 (range: 0.8732 - 0.9182)
- **Hit@10:** 0.9203 ± 0.0139 (range: 0.8991 - 0.9360)


### Model Type Comparison

**Neural Models (3 models):**
- Recall@10: 0.9203
- MRR@10: 0.8944
- NDCG@10: 0.9007


## Statistical Analysis

### Variance Analysis

**Recall@10:**
- Between-model variance: 0.000261
- Within-model variance: 0.000007
- Ratio: 35.77x

**MRR@10:**
- Between-model variance: 0.000334
- Within-model variance: 0.000031
- Ratio: 10.61x


---

**Generated:** 2025-12-16 10:11:43  
**Total Duration:** See experiment logs for detailed timing
