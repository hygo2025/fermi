# Experiment Results - 2025-12-16 11:03:39

## Execution Summary

- **Timestamp:** 2025-12-16 11:03:39
- **Total Models:** 1
- **Total Slices:** 1
- **Total Experiments:** 1 (1 models × 1 slices)
- **Metrics Evaluated:** 12 (Recall, MRR, NDCG, Hit @ K=5,10,20)

## Best Models by Metric

- **Recall@10:** GRU4Rec (`0.9237±nan`)
- **MRR@10:** GRU4Rec (`0.9009±nan`)
- **NDCG@10:** GRU4Rec (`0.9064±nan`)
- **Hit@10:** GRU4Rec (`0.9237±nan`)


## Models Evaluated

### [Neural] GRU4Rec
- Experiments: 1
- Slices: [1]
- **Recall@10:** 0.9237±nan
- **MRR@10:** 0.9009±nan
- **NDCG@10:** 0.9064±nan
- **Hit@10:** 0.9237±nan

## Temporal Slices

### Slice 1
- Models evaluated: 1
- Best model: **GRU4Rec** (Recall@10: 0.9237)
- Mean Recall@10: 0.9237 ± nan
- Mean MRR@10: 0.9009 ± nan

## Top 3 Models (by Recall@10)

### 1. GRU4Rec

| Metric | @5 | @10 | @20 |
|--------|-----|-----|-----|
| **Recall** | 0.9170±nan | 0.9237±nan | 0.9305±nan |
| **MRR** | 0.9000±nan | 0.9009±nan | 0.9014±nan |
| **NDCG** | 0.9043±nan | 0.9064±nan | 0.9082±nan |
| **Hit** | 0.9170±nan | 0.9237±nan | 0.9305±nan |

## Generated Files

- `raw_results.csv` - Complete results for all 1 experiments
- `aggregated_results.csv` - Aggregated metrics (mean ± std across slices)
- `results_table.md` - Markdown table with key metrics
- `metrics_comparison.png` - Bar chart comparing models (with error bars)
- `performance_heatmap.png` - Normalized performance heatmap
- `slice_consistency.png` - Performance consistency across temporal slices

## Methodology

### Data Split
- **Temporal Slices:** 1 slices using sliding window protocol
- **Protocol:** Next-item prediction
- **Evaluation:** Leave-one-out on test set

### Metrics Aggregation
- **Mean:** Average performance across 1 temporal slices
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
- **Recall@10:** 0.9237 ± nan (range: 0.9237 - 0.9237)
- **MRR@10:** 0.9009 ± nan (range: 0.9009 - 0.9009)
- **NDCG@10:** 0.9064 ± nan (range: 0.9064 - 0.9064)
- **Hit@10:** 0.9237 ± nan (range: 0.9237 - 0.9237)


### Model Type Comparison

**Neural Models (1 models):**
- Recall@10: 0.9237
- MRR@10: 0.9009
- NDCG@10: 0.9064


## Statistical Analysis

### Variance Analysis

**Recall@10:**
- Between-model variance: nan
- Within-model variance: nan
- Ratio: nanx

**MRR@10:**
- Between-model variance: nan
- Within-model variance: nan
- Ratio: nanx


---

**Generated:** 2025-12-16 11:03:39  
**Total Duration:** See experiment logs for detailed timing
