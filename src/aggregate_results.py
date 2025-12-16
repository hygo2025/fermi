import argparse
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ResultsAggregator:
    """Aggregate experimental results and generate publication-ready tables/plots"""
    
    def __init__(self, input_path: str, output_dir: str, create_timestamped: bool = True):
        self.input_path = Path(input_path)
        
        # Create timestamped directory if requested
        if create_timestamped:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Input path is outputs/results, so aggregated goes inside it
            self.output_dir = self.input_path / 'aggregated' / timestamp
        else:
            self.output_dir = Path(output_dir).parent
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now()
        
        # Set publication-quality plot style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
    def load_results(self) -> pd.DataFrame:
        input_file = self.input_path / 'raw_results.csv'
        
        if not input_file.exists():
            raise FileNotFoundError(f"{input_file} not found")
        
        df = pd.read_csv(input_file)
        
        # Normalize metric column names to Title Case (Recall@10, not recall@10)
        new_columns = []
        for col in df.columns:
            if '@' in col:
                col = (col.replace('recall@', 'Recall@')
                          .replace('mrr@', 'MRR@')
                          .replace('ndcg@', 'NDCG@')
                          .replace('hit@', 'Hit@'))
            new_columns.append(col)
        df.columns = new_columns
        
        # Remove duplicates (same model + slice)
        initial_count = len(df)
        df = df.drop_duplicates(subset=['model', 'slice'], keep='last')
        if len(df) < initial_count:
            print(f"WARNING: Removed {initial_count - len(df)} duplicate entries")
        
        print(f"Loaded {len(df)} results from {input_file}")
        print(f"Models: {sorted(df['model'].unique())}")
        print(f"Slices: {sorted(df['slice'].unique())}")
        
        return df
    
    def aggregate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula mean ± std por modelo (estilo Domingues et al. 2024)"""
        
        # Find metric columns
        metric_cols = [col for col in df.columns if '@' in col]
        print(f"\nMetrics found: {metric_cols}")
        
        aggregated = []
        for model in sorted(df['model'].unique()):
            model_data = df[df['model'] == model]
            row = {'Model': model}
            
            for metric in metric_cols:
                values = model_data[metric].values
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)  # Sample std
                
                # Format como no paper: mean ± std
                row[metric] = f"{mean_val:.4f}±{std_val:.4f}"
                row[f'{metric}_mean'] = mean_val
                row[f'{metric}_std'] = std_val
            
            aggregated.append(row)
        
        return pd.DataFrame(aggregated)
    
    def create_latex_table(self, df: pd.DataFrame) -> str:
        """Cria tabela LaTeX estilo paper"""
        
        # Select main metrics @10 and @20
        display_metrics = [col for col in df.columns 
                          if '@10' in col or '@20' in col]
        display_metrics = [m for m in display_metrics if not m.endswith('_mean') and not m.endswith('_std')]
        
        table_df = df[['Model'] + display_metrics].copy()
        
        # Rename metrics for cleaner display
        rename_map = {col: col.replace('Recall@', 'R@').replace('MRR@', 'M@')
                     .replace('NDCG@', 'N@').replace('Hit@', 'H@')
                     for col in display_metrics}
        table_df = table_df.rename(columns=rename_map)
        
        # Generate LaTeX
        latex = table_df.to_latex(index=False, escape=False, float_format="%.4f")
        
        return latex
    
    def plot_metrics_comparison(self, df: pd.DataFrame):
        """Gráfico de comparação de modelos (estilo paper)"""
        
        metrics = ['Recall@10', 'MRR@10', 'NDCG@10', 'Hit@10']
        available_metrics = [m for m in metrics if f'{m}_mean' in df.columns]
        
        if not available_metrics:
            print("No metrics found for plotting")
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 4))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            models = df['Model'].values
            means = df[f'{metric}_mean'].values
            stds = df[f'{metric}_std'].values
            
            # Bar plot with error bars
            x_pos = np.arange(len(models))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                         alpha=0.8, edgecolor='black', linewidth=1.2)
            
            # Color best model
            best_idx = np.argmax(means)
            bars[best_idx].set_color('#2ecc71')
            
            ax.set_xlabel('Model', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
        
        plt.tight_layout()
        output_file = self.output_dir / 'metrics_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_file}")
        plt.close()
    
    def plot_metrics_heatmap(self, df: pd.DataFrame):
        """Heatmap de performance (estilo paper)"""
        
        # Get mean values
        mean_cols = [col for col in df.columns if col.endswith('_mean')]
        
        if not mean_cols:
            print("No mean metrics for heatmap")
            return
        
        # Create matrix
        matrix_df = df[['Model'] + mean_cols].set_index('Model')
        
        # Clean column names
        matrix_df.columns = [col.replace('_mean', '') for col in matrix_df.columns]
        
        # Normalize to [0, 1] for better visualization
        matrix_norm = (matrix_df - matrix_df.min()) / (matrix_df.max() - matrix_df.min())
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, len(df)*0.8))
        sns.heatmap(matrix_norm.T, annot=matrix_df.T.values, fmt='.4f',
                   cmap='RdYlGn', vmin=0, vmax=1, 
                   cbar_kws={'label': 'Normalized Score'},
                   linewidths=1, linecolor='gray', ax=ax)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_file = self.output_dir / 'performance_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap: {output_file}")
        plt.close()
    
    def plot_slice_consistency(self, raw_df: pd.DataFrame):
        """Gráfico de consistência entre slices (análise temporal)"""
        
        metric = 'Recall@10'
        if metric not in raw_df.columns:
            print(f"Metric {metric} not found for slice analysis")
            return
        
        # Pivot data
        pivot_df = raw_df.pivot(index='slice', columns='model', values=metric)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model in pivot_df.columns:
            ax.plot(pivot_df.index, pivot_df[model], marker='o', 
                   linewidth=2, markersize=8, label=model, alpha=0.8)
        
        ax.set_xlabel('Slice', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} across Temporal Slices', fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        output_file = self.output_dir / 'slice_consistency.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved slice consistency plot: {output_file}")
        plt.close()
    
    def organize_model_files(self):
        """Organiza arquivos específicos de cada modelo em suas pastas"""
        import shutil
        import json
        
        print("\nOrganizing model-specific files...")
        
        # Get all models from results
        raw_results = self.input_path / 'raw_results.csv'
        if not raw_results.exists():
            return
        
        df = pd.read_csv(raw_results)
        models = df['model'].unique()
        
        # Create model folders
        for model in models:
            model_dir = self.output_dir / model
            model_dir.mkdir(exist_ok=True)
            
            # Move loss files
            losses_dir = self.input_path / 'losses'
            if losses_dir.exists():
                for loss_file in losses_dir.glob(f'{model}_slice*_loss.json'):
                    dest = model_dir / loss_file.name
                    shutil.copy2(loss_file, dest)
            
            # Move execution logs
            for log_file in self.input_path.glob(f'{model}_*.log'):
                dest = model_dir / log_file.name
                shutil.copy2(log_file, dest)
            
            # Move model-specific checkpoints if any
            saved_dir = self.input_path / 'outputs' / 'saved'
            if saved_dir.exists():
                for model_file in saved_dir.glob(f'{model}*'):
                    if model_file.is_file():
                        dest = model_dir / model_file.name
                        shutil.copy2(model_file, dest)
        
        print(f"Organized files for {len(models)} models into separate folders")
    
    def plot_loss_curves(self):
        """Gráficos de training loss curves"""
        
        losses_dir = self.input_path / 'losses'
        
        if not losses_dir.exists():
            print("No loss data found. Skipping loss curves...")
            return
        
        import json
        import glob
        
        # Load all loss files
        loss_files = list(losses_dir.glob('*_loss.json'))
        
        if not loss_files:
            print("No loss files found")
            return
        
        # Group by model
        model_losses = {}
        for loss_file in loss_files:
            with open(loss_file, 'r') as f:
                data = json.load(f)
            
            # Extract model and slice from filename: model_sliceN_loss.json
            filename = loss_file.stem  # removes .json
            parts = filename.split('_')
            model = '_'.join(parts[:-2])  # Everything except 'sliceN' and 'loss'
            slice_id = parts[-2].replace('slice', '')
            
            if model not in model_losses:
                model_losses[model] = []
            
            model_losses[model].append({
                'slice': int(slice_id),
                'train_losses': data.get('train_losses', [])
            })
        
        # Create plots for each model and save in model folder
        for model, slices_data in model_losses.items():
            # Create model directory
            model_dir = self.output_dir / model
            model_dir.mkdir(exist_ok=True)
            
            n_slices = len(slices_data)
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            
            fig.suptitle(f'{model} - Training History (All Slices)', 
                        fontsize=14, fontweight='bold')
            
            for idx, slice_data in enumerate(sorted(slices_data, key=lambda x: x['slice'])):
                if idx >= 6:  # Max 6 slices to fit in 2x3 grid
                    break
                
                ax = axes[idx]
                slice_id = slice_data['slice']
                train_losses = slice_data['train_losses']
                
                epochs = list(range(1, len(train_losses) + 1))
                
                # Plot training loss
                if train_losses:
                    ax.plot(epochs, train_losses, 'b-', label='Train Loss', 
                           linewidth=2, alpha=0.7)
                
                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel('Training Loss', fontsize=10)
                ax.set_title(f'Slice {slice_id}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(loc='best', fontsize=8)
            
            # Hide unused subplots
            for idx in range(n_slices, 6):
                axes[idx].axis('off')
            
            plt.tight_layout()
            output_file = model_dir / 'loss_curves.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved loss curves: {output_file}")
            plt.close()
        
        # Create summary plot - average loss across slices
        self.plot_average_loss_curves(model_losses)
    
    
    def plot_average_loss_curves(self, model_losses):
        """Plot average training loss curves across all slices for comparison"""
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for model, slices_data in model_losses.items():
            # Calculate average train loss across slices
            max_epochs = max(len(s['train_losses']) for s in slices_data if s['train_losses'])
            
            if max_epochs == 0:
                continue
            
            avg_train_loss = []
            
            for epoch in range(max_epochs):
                train_vals = [s['train_losses'][epoch] for s in slices_data 
                             if len(s['train_losses']) > epoch]
                if train_vals:
                    avg_train_loss.append(np.mean(train_vals))
            
            # Plot average training loss
            if avg_train_loss:
                epochs = list(range(1, len(avg_train_loss) + 1))
                ax.plot(epochs, avg_train_loss, marker='o', linewidth=2, 
                        label=model, alpha=0.8, markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Training Loss', fontsize=12, fontweight='bold')
        ax.set_title('Training Loss (Average across slices)', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_file = self.output_dir / 'loss_curves_average.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved average loss curves: {output_file}")
        plt.close()
    
    def create_summary_tables(self, agg_df: pd.DataFrame):
        """Cria tabelas resumidas em diferentes formatos"""
        
        # 1. CSV completo
        csv_file = self.output_dir / 'aggregated_results.csv'
        agg_df.to_csv(csv_file, index=False)
        print(f"Saved CSV: {csv_file}")
        
        # 2. Markdown table (for GitHub/docs)
        md_file = self.output_dir / 'results_table.md'
        
        # Select key metrics
        key_metrics = [col for col in agg_df.columns 
                      if '@10' in col and not col.endswith('_mean') and not col.endswith('_std')]
        md_df = agg_df[['Model'] + key_metrics]
        
        with open(md_file, 'w') as f:
            f.write("# Experiment Results\n\n")
            f.write(md_df.to_markdown(index=False))
            f.write("\n\n*Values shown as mean ± std across 5 temporal slices*\n")
        print(f"Saved Markdown table: {md_file}")
    
    def create_execution_readme(self, raw_df: pd.DataFrame, agg_df: pd.DataFrame):
        """Cria README com informações detalhadas da execução"""
        
        readme_file = self.output_dir / 'README.md'
        
        # Collect statistics
        n_models = len(agg_df)
        n_slices = len(raw_df['slice'].unique())
        models_list = sorted(raw_df['model'].unique())
        slices_list = sorted(raw_df['slice'].unique())
        
        # Find best model for each metric
        metrics_to_check = ['Recall@10', 'MRR@10', 'NDCG@10', 'Hit@10']
        best_models = {}
        for metric in metrics_to_check:
            if f'{metric}_mean' in agg_df.columns:
                best_idx = agg_df[f'{metric}_mean'].idxmax()
                best_models[metric] = {
                    'model': agg_df.loc[best_idx, 'Model'],
                    'value': agg_df.loc[best_idx, metric]
                }
        
        # Calculate overall statistics
        all_metrics = [col for col in raw_df.columns if '@' in col]
        stats_per_model = {}
        for model in models_list:
            model_data = raw_df[raw_df['model'] == model]
            stats_per_model[model] = {
                'n_experiments': len(model_data),
                'slices': sorted(model_data['slice'].unique())
            }
        
        content = f"""# Experiment Results - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Execution Summary

- **Timestamp:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- **Total Models:** {n_models}
- **Total Slices:** {n_slices}
- **Total Experiments:** {len(raw_df)} ({n_models} models × {n_slices} slices)
- **Metrics Evaluated:** {len(all_metrics)} (Recall, MRR, NDCG, Hit @ K=5,10,20)

## Best Models by Metric

"""
        for metric, info in best_models.items():
            content += f"- **{metric}:** {info['model']} (`{info['value']}`)\n"
        
        content += f"""

## Models Evaluated

"""
        for model in models_list:
            stats = stats_per_model[model]
            model_type = "[Neural]" if model in ['GRU4Rec', 'NARM', 'STAMP', 'SASRec'] else "[Baseline]"
            content += f"### {model_type} {model}\n"
            content += f"- Experiments: {stats['n_experiments']}\n"
            content += f"- Slices: {stats['slices']}\n"
            
            # Get model performance summary
            model_perf = agg_df[agg_df['Model'] == model].iloc[0]
            content += f"- **Recall@10:** {model_perf['Recall@10']}\n"
            content += f"- **MRR@10:** {model_perf['MRR@10']}\n"
            content += f"- **NDCG@10:** {model_perf['NDCG@10']}\n"
            content += f"- **Hit@10:** {model_perf['Hit@10']}\n\n"
        
        content += f"""## Temporal Slices

"""
        for slice_id in slices_list:
            slice_data = raw_df[raw_df['slice'] == slice_id]
            content += f"### Slice {slice_id}\n"
            content += f"- Models evaluated: {len(slice_data)}\n"
            
            # Best model in this slice
            best_model_slice = slice_data.loc[slice_data['Recall@10'].idxmax()]
            content += f"- Best model: **{best_model_slice['model']}** (Recall@10: {best_model_slice['Recall@10']:.4f})\n"
            
            # Statistics for this slice
            content += f"- Mean Recall@10: {slice_data['Recall@10'].mean():.4f} ± {slice_data['Recall@10'].std():.4f}\n"
            content += f"- Mean MRR@10: {slice_data['MRR@10'].mean():.4f} ± {slice_data['MRR@10'].std():.4f}\n\n"
        
        content += f"""## Top 3 Models (by Recall@10)

"""
        # Add top 3 models with all metrics
        top3 = agg_df.nlargest(3, 'Recall@10_mean')
        for rank, (idx, row) in enumerate(top3.iterrows(), start=1):
            content += f"### {rank}. {row['Model']}\n\n"
            content += "| Metric | @5 | @10 | @20 |\n"
            content += "|--------|-----|-----|-----|\n"
            
            for metric_base in ['Recall', 'MRR', 'NDCG', 'Hit']:
                values = []
                for k in [5, 10, 20]:
                    col = f'{metric_base}@{k}'
                    if col in row:
                        values.append(row[col])
                    else:
                        values.append('N/A')
                content += f"| **{metric_base}** | {values[0]} | {values[1]} | {values[2]} |\n"
            content += "\n"
        
        content += f"""## Generated Files

- `raw_results.csv` - Complete results for all {len(raw_df)} experiments
- `aggregated_results.csv` - Aggregated metrics (mean ± std across slices)
- `results_table.md` - Markdown table with key metrics
- `metrics_comparison.png` - Bar chart comparing models (with error bars)
- `performance_heatmap.png` - Normalized performance heatmap
- `slice_consistency.png` - Performance consistency across temporal slices

## Methodology

### Data Split
- **Temporal Slices:** {n_slices} slices using sliding window protocol
- **Protocol:** Next-item prediction
- **Evaluation:** Leave-one-out on test set

### Metrics Aggregation
- **Mean:** Average performance across {n_slices} temporal slices
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
"""
        
        # Add overall statistics
        for metric in ['Recall@10', 'MRR@10', 'NDCG@10', 'Hit@10']:
            if metric in raw_df.columns:
                mean_val = raw_df[metric].mean()
                std_val = raw_df[metric].std()
                min_val = raw_df[metric].min()
                max_val = raw_df[metric].max()
                content += f"- **{metric}:** {mean_val:.4f} ± {std_val:.4f} (range: {min_val:.4f} - {max_val:.4f})\n"
        
        content += f"""

### Model Type Comparison
"""
        # Compare neural vs baseline
        neural_models = ['GRU4Rec', 'NARM', 'STAMP', 'SASRec']
        baseline_models = [m for m in models_list if m not in neural_models]
        
        if neural_models:
            neural_data = raw_df[raw_df['model'].isin(neural_models)]
            content += f"\n**Neural Models ({len([m for m in neural_models if m in models_list])} models):**\n"
            for metric in ['Recall@10', 'MRR@10', 'NDCG@10']:
                if metric in neural_data.columns:
                    mean_val = neural_data[metric].mean()
                    content += f"- {metric}: {mean_val:.4f}\n"
        
        if baseline_models:
            baseline_data = raw_df[raw_df['model'].isin(baseline_models)]
            content += f"\n**Baseline Models ({len(baseline_models)} models):**\n"
            for metric in ['Recall@10', 'MRR@10', 'NDCG@10']:
                if metric in baseline_data.columns:
                    mean_val = baseline_data[metric].mean()
                    content += f"- {metric}: {mean_val:.4f}\n"
        
        content += f"""

## Statistical Analysis

### Variance Analysis
"""
        # Variance analysis
        for metric in ['Recall@10', 'MRR@10']:
            if metric in raw_df.columns:
                # Variance between models
                model_means = raw_df.groupby('model')[metric].mean()
                model_variance = model_means.var()
                
                # Variance within models (across slices)
                within_variance = raw_df.groupby('model')[metric].var().mean()
                
                content += f"\n**{metric}:**\n"
                content += f"- Between-model variance: {model_variance:.6f}\n"
                content += f"- Within-model variance: {within_variance:.6f}\n"
                content += f"- Ratio: {model_variance/within_variance:.2f}x\n"
        
        content += f"""

---

**Generated:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  
**Total Duration:** See experiment logs for detailed timing
"""
        
        with open(readme_file, 'w') as f:
            f.write(content)
        
        print(f"Saved execution README: {readme_file}")
    
    def run(self):
        """Executa pipeline completo de agregação e visualização"""
        
        print("="*80)
        print("Results Aggregation & Visualization Pipeline")
        print("="*80)
        print()
        
        # Load data
        raw_df = self.load_results()
        
        # Aggregate
        print("\nAggregating metrics...")
        agg_df = self.aggregate_metrics(raw_df)
        
        # Sort by best Recall@10
        if 'Recall@10_mean' in agg_df.columns:
            agg_df = agg_df.sort_values('Recall@10_mean', ascending=False)
        
        # Display in terminal
        print("\n" + "="*80)
        print("AGGREGATED RESULTS (mean ± std)")
        print("="*80)
        display_cols = ['Model'] + [col for col in agg_df.columns 
                                    if '@' in col and not col.endswith('_mean') 
                                    and not col.endswith('_std')]
        print(agg_df[display_cols].to_string(index=False))
        print()
        
        # Organize model-specific files into folders
        self.organize_model_files()
        
        # Create tables
        print("\nGenerating tables...")
        self.create_summary_tables(agg_df)
        self.create_execution_readme(raw_df, agg_df)
        
        # Create plots
        print("\nGenerating plots...")
        self.plot_metrics_comparison(agg_df)
        self.plot_metrics_heatmap(agg_df)
        self.plot_slice_consistency(raw_df)
        self.plot_loss_curves()  # Training loss curves
        
        print("\n" + "="*80)
        print(f"All outputs saved to: {self.output_dir}")
        print("="*80)
        print("\nGenerated files:")
        for f in sorted(self.output_dir.glob('*')):
            if f.is_file():
                print(f"  - {f.name}")
            else:
                print(f"  - {f.name}/ (model-specific files)")
        print("\nModel-specific folders contain:")
        print("  - Loss curves (loss_curves.png)")
        print("  - Loss data (*.json)")
        print("  - Execution logs (*.log)")
        print("  - Model checkpoints (if saved)")


def aggregate_results(input_path: str, output_path: str):
    """Legacy function for backward compatibility"""
    output_dir = Path(output_path).parent
    aggregator = ResultsAggregator(input_path, output_dir)
    aggregator.run()


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate experiment results and generate tables/plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python aggregate_results.py --input outputs/results --output outputs/results/aggregated_results.csv
  
  # Using make
  make aggregate-results
  
Outputs:
  - aggregated_results.csv      # CSV with mean ± std
  - results_table.md            # Markdown table for README
  - metrics_comparison.png      # Bar plots
  - performance_heatmap.png     # Heatmap visualization
  - slice_consistency.png       # Temporal consistency plot
        """
    )
    parser.add_argument('--input', type=str, default='outputs/results',
                       help='Input directory with raw results')
    parser.add_argument('--output', type=str, default='outputs/results/aggregated_results.csv',
                       help='Output CSV file (other formats auto-generated)')
    
    args = parser.parse_args()
    
    aggregate_results(args.input, args.output)


if __name__ == '__main__':
    main()
