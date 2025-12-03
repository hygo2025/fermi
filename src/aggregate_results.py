#!/usr/bin/env python3
"""
Aggregate Results - Compute mean ± std across slices

Processa resultados brutos e gera tabelas agregadas no formato do paper
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def aggregate_results(input_path: str, output_path: str):
    """
    Agrega resultados de todos os slices
    
    Args:
        input_path: Diretório com resultados brutos
        output_path: Arquivo de saída (.csv)
    """
    input_file = Path(input_path) / 'raw_results.csv'
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return
    
    # Load raw results
    df = pd.read_parquet(input_file)
    
    print(f"Loaded {len(df)} results from {input_file}")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Slices: {df['slice'].unique().tolist()}")
    
    # Find metric columns
    metric_cols = [col for col in df.columns if '@' in col]
    
    print(f"\nMetrics found: {metric_cols}")
    
    # Group by model
    grouped = df.groupby('model')
    
    # Calculate mean ± std
    aggregated = []
    for model, group in grouped:
        row = {'model': model, 'n_slices': len(group)}
        
        for metric in metric_cols:
            if metric in group.columns:
                values = group[metric].values
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                row[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
                row[f'{metric}_mean'] = mean_val
                row[f'{metric}_std'] = std_val
        
        aggregated.append(row)
    
    # Create DataFrame
    agg_df = pd.DataFrame(aggregated)
    
    # Sort by a key metric (e.g., Recall@10_mean)
    if 'Recall@10_mean' in agg_df.columns:
        agg_df = agg_df.sort_values('Recall@10_mean', ascending=False)
    
    # Save
    output_file = Path(output_path)
    agg_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"Aggregated results saved to {output_file}")
    print(f"{'='*80}")
    
    # Display summary
    display_cols = ['model', 'n_slices'] + [col for col in agg_df.columns if '@' in col and 'mean' not in col and 'std' not in col]
    if display_cols:
        print("\nAggregated Results (mean ± std):\n")
        print(agg_df[display_cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Aggregate experiment results')
    parser.add_argument('--input', type=str, default='results',
                       help='Input directory with raw results')
    parser.add_argument('--output', type=str, default='results/aggregated_results.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    aggregate_results(args.input, args.output)


if __name__ == '__main__':
    main()
