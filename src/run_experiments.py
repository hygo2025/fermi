#!/usr/bin/env python3
"""
Experiment Runner - Session-Based Recommendation Benchmark

Executa modelos RecBole em todos os slices do sliding window e agrega resultados
Seguindo metodologia de Domingues et al. (2024)
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec, NARM, STAMP, SASRec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, get_model

from metrics import SessionBasedMetrics


class ExperimentRunner:
    """Executa experimentos de recomendação em múltiplos slices"""
    
    def __init__(self, 
                 data_path: str = 'recbole_data',
                 output_path: str = 'results',
                 config_path: str = 'src/configs',
                 models: List[str] = None,
                 slices: List[int] = None):
        """
        Args:
            data_path: Diretório com dados RecBole
            output_path: Diretório para salvar resultados
            config_path: Diretório com configs YAML dos modelos
            models: Lista de modelos para executar (None = todos)
            slices: Lista de slices para processar (None = todos)
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.config_path = Path(config_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Modelos disponíveis
        self.available_models = {
            'GRU4Rec': GRU4Rec,
            'NARM': NARM,
            'STAMP': STAMP,
            'SASRec': SASRec,
        }
        
        self.models = models if models else list(self.available_models.keys())
        
        # Find available slices
        if slices is None:
            self.slices = sorted([
                int(p.name.replace('realestate_slice', ''))
                for p in self.data_path.glob('realestate_slice*')
                if p.is_dir()
            ])
        else:
            self.slices = slices
        
        # Métricas
        self.metrics = SessionBasedMetrics(k_values=[5, 10, 20])
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Configura logging"""
        log_dir = self.output_path / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_model_config(self, model_name: str, dataset_name: str) -> Dict:
        """
        Carrega configuração do modelo do arquivo YAML
        
        Args:
            model_name: Nome do modelo
            dataset_name: Nome do dataset (realestate_slice{N})
            
        Returns:
            Dict com configurações
        """
        # Load base config from YAML file
        config_file = self.config_path / 'neural' / f'{model_name.lower()}.yaml'
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override dataset and data_path for current slice
        config['dataset'] = dataset_name
        config['data_path'] = str(self.data_path)
        
        return config
    
    def run_single_experiment(self, model_name: str, slice_id: int) -> Dict:
        """
        Executa um experimento: um modelo em um slice
        
        Args:
            model_name: Nome do modelo
            slice_id: ID do slice
            
        Returns:
            Dict com resultados
        """
        dataset_name = f'realestate_slice{slice_id}'
        self.logger.info(f"Running {model_name} on {dataset_name}...")
        
        # Get config
        config_dict = self.get_model_config(model_name, dataset_name)
        config = Config(model=model_name, config_dict=config_dict)
        
        # Init seed
        init_seed(config['seed'], config['reproducibility'])
        
        # Load dataset
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        
        # Calculate item popularity from train data
        train_items = []
        for batch in train_data:
            train_items.extend(batch[config['ITEM_ID_FIELD']].cpu().numpy().tolist())
        self.metrics.set_item_popularity(train_items)
        
        # Create model
        model_class = self.available_models[model_name]
        model = model_class(config, train_data.dataset).to(config['device'])
        
        # Create trainer
        trainer = Trainer(config, model)
        
        # Train
        self.logger.info(f"  Training {model_name}...")
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, show_progress=True
        )
        
        # Test
        self.logger.info(f"  Testing {model_name}...")
        test_result = trainer.evaluate(test_data, show_progress=True)
        
        # Format results
        results = {
            'model': model_name,
            'slice': slice_id,
            'dataset': dataset_name,
            'best_valid_score': best_valid_score,
            **test_result
        }
        
        self.logger.info(f"  Results: {test_result}")
        
        return results
    
    def run_all_experiments(self):
        """Executa todos os experimentos (todos modelos em todos slices)"""
        self.logger.info("="*80)
        self.logger.info("Starting Session-Based Recommendation Experiments")
        self.logger.info("="*80)
        self.logger.info(f"Models: {self.models}")
        self.logger.info(f"Slices: {self.slices}")
        self.logger.info("")
        
        all_results = []
        
        for model_name in self.models:
            for slice_id in self.slices:
                try:
                    result = self.run_single_experiment(model_name, slice_id)
                    all_results.append(result)
                    
                    # Save intermediate results
                    self.save_results(all_results)
                    
                except Exception as e:
                    self.logger.error(f"Error running {model_name} on slice {slice_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        self.logger.info("="*80)
        self.logger.info("All experiments completed!")
        self.logger.info("="*80)
        
        return all_results
    
    def save_results(self, results: List[Dict]):
        """Salva resultados em CSV"""
        df = pd.DataFrame(results)
        output_file = self.output_path / 'raw_results.csv'
        df.to_csv(output_file, index=False)
        self.logger.info(f"Results saved to {output_file}")
    
    def aggregate_results(self, results: List[Dict]) -> pd.DataFrame:
        """
        Agrega resultados por modelo (média ± std entre slices)
        
        Args:
            results: Lista de dicts com resultados
            
        Returns:
            DataFrame agregado
        """
        if not results:
            self.logger.warning("No results to aggregate!")
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Group by model
        grouped = df.groupby('model')
        
        # Calculate mean and std for each metric
        metric_cols = [col for col in df.columns if '@' in col]
        
        aggregated = []
        for model, group in grouped:
            row = {'model': model}
            for metric in metric_cols:
                if metric in group.columns:
                    values = group[metric].values
                    row[f'{metric}_mean'] = np.mean(values)
                    row[f'{metric}_std'] = np.std(values)
            aggregated.append(row)
        
        agg_df = pd.DataFrame(aggregated)
        
        # Save
        output_file = self.output_path / 'aggregated_results.csv'
        agg_df.to_csv(output_file, index=False)
        self.logger.info(f"Aggregated results saved to {output_file}")
        
        return agg_df


def main():
    parser = argparse.ArgumentParser(description='Run session-based recommendation experiments')
    parser.add_argument('--data-path', type=str, default='recbole_data',
                       help='Path to RecBole data')
    parser.add_argument('--output-path', type=str, default='results',
                       help='Path to save results')
    parser.add_argument('--config-path', type=str, default='src/configs',
                       help='Path to model config files')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Models to run (default: all)')
    parser.add_argument('--slices', type=int, nargs='+', default=None,
                       help='Slices to process (default: all)')
    parser.add_argument('--all-slices', action='store_true',
                       help='Run on all slices')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        data_path=args.data_path,
        output_path=args.output_path,
        config_path=args.config_path,
        models=args.models,
        slices=args.slices
    )
    
    results = runner.run_all_experiments()
    runner.aggregate_results(results)


if __name__ == '__main__':
    main()
