"""
Hyperparameter Tuning for RecBole Models

This script implements hyperparameter search using RecBole's HyperTuning.
Reference: https://github.com/RUCAIBox/RecBole/blob/master/docs/source/user_guide/usage/parameter_tuning.rst

Usage:
    python src/hyperparameter_tuning.py --model GRU4Rec --max-evals 20
    python src/hyperparameter_tuning.py --model GRU4Rec --algo bayes --max-evals 50
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from hyperopt import hp
import torch
import yaml
from recbole.quick_start import objective_function, run_recbole
from recbole.trainer import HyperTuning

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def custom_objective_function(config_dict, config_file_list, saved=True):
    """
    Wrapper para garantir que hiperparâmetros numéricos sejam passados
    como int/float corretos, evitando erro de str no PyTorch.
    """
    # Lista de parâmetros que DEVEM ser inteiros
    int_params = [
        'embedding_size', 'hidden_size', 'num_layers',
        'epochs', 'stopping_step', 'max_evals',
        'train_batch_size', 'eval_batch_size', 'topk'
    ]

    for param in int_params:
        if param in config_dict:
            val = config_dict[param]
            # Se for string, tenta converter
            if isinstance(val, str):
                try:
                    # Tenta converter para int (lida com '64')
                    config_dict[param] = int(val)
                except ValueError:
                    # Se falhar (ex: '[5, 10]'), deixa como está
                    pass
            # Se for float mas deveria ser int (ex: 64.0 -> 64)
            elif isinstance(val, float) and val.is_integer():
                config_dict[param] = int(val)

    return objective_function(config_dict, config_file_list, saved)

class HyperparameterTuner:
    """Hyperparameter tuning for RecBole models"""
    
    def __init__(self, 
                 model_name: str,
                 dataset_name: str = 'realestate_simple',
                 data_path: str = 'outputs/data/recbole_simple',
                 output_path: str = 'outputs/tuning',
                 gpu_id: int = 0):
        """
        Args:
            model_name: Model to tune (e.g., 'GRU4Rec', 'NARM', 'SASRec')
            dataset_name: Dataset name
            data_path: Path to RecBole data
            output_path: Where to save tuning results
            gpu_id: GPU ID to use
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.gpu_id = gpu_id
        
        logger.info(f"Initializing HyperparameterTuner for {model_name}")
        logger.info(f"  Dataset: {dataset_name}")
        logger.info(f"  Data path: {self.data_path}")
        logger.info(f"  Output path: {self.output_path}")
        logger.info(f"  GPU: {gpu_id}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        
    def get_base_config(self) -> dict:
        """Get base configuration (fixed parameters)"""
        config = {
            'model': self.model_name,
            'dataset': self.dataset_name,
            'data_path': str(self.data_path),
            
            # Training settings
            'epochs': 10,
            'train_batch_size': 2048,
            'eval_batch_size': 2048,
            'train_neg_sample_args': None,
            
            # Early Stopping
            'stopping_step': 2,
            
            # Fixed Model Parameters
            'loss_type': 'CE',
            'num_layers': 2,
            
            # Gradient Clipping
            'clip_grad_norm': {'max_norm': 5.0},

            # Evaluation
            'metrics': ['Recall', 'MRR', 'NDCG', 'Hit'],
            'topk': [5, 10, 20],
            'valid_metric': 'MRR@10',

            # Session Settings
            'MAX_ITEM_LIST_LENGTH': 50,
            'SESSION_ID_FIELD': 'session_id',
            'ITEM_ID_FIELD': 'item_id',
            'TIME_FIELD': 'timestamp',
            'USER_ID_FIELD': 'session_id',
            'load_col': {'inter': ['session_id', 'item_id', 'timestamp']},

            # Evaluation Protocol
            'eval_args': {
                'split': {'LS': 'valid_and_test'},
                'order': 'TO',
                'mode': 'full'
            },
            
            # Device
            'device': 'cuda',
            'gpu_id': self.gpu_id,
            
            # Reproducibility
            'seed': 42,
            
            # Checkpointing
            'save_dataset': False,
            'save_dataloaders': False,
            'checkpoint_dir': str(self.output_path / 'checkpoints'),
            
            # Logging
            'log_wandb': False,
        }
        
        return config
    
    def get_search_space(self) -> dict:
        """
        Get hyperparameter search space
        
        Format: {'param_name': 'distribution(args)'}
        
        Supported distributions:
        - choice([val1, val2, ...]): discrete choice
        - uniform(low, high): uniform distribution
        - loguniform(low, high): log-uniform distribution
        - randint(low, high): random integer
        """
        space = {
            # Learning rate (log scale)
            'learning_rate': 'loguniform(1e-4, 1e-2)',
            
            # Embedding dimension
            'embedding_size': 'choice([32, 64, 128])',
            
            # Hidden dimension
            'hidden_size': 'choice([64, 128, 256])',
            
            # Dropout probability
            'dropout_prob': 'uniform(0.1, 0.5)',
        }
        
        logger.info("\nHyperparameter search space:")
        for param, dist in space.items():
            logger.info(f"  {param}: {dist}")
        
        return space

    def _parse_search_space(self, space_dict: dict) -> dict:
        """Parse string search space to hyperopt objects"""
        parsed_space = {}
        for param, value in space_dict.items():
            if not isinstance(value, str):
                parsed_space[param] = value
                continue
                
            if value.startswith('choice'):
                # choice([32, 64, 128])
                content = value[value.find('(')+1 : value.rfind(')')]
                choices = eval(content)
                parsed_space[param] = hp.choice(param, choices)
            elif value.startswith('loguniform'):
                # loguniform(1e-4, 1e-2)
                content = value[value.find('(')+1 : value.rfind(')')]
                low, high = map(float, content.split(','))
                parsed_space[param] = hp.loguniform(param, np.log(low), np.log(high))
            elif value.startswith('uniform'):
                # uniform(0.1, 0.5)
                content = value[value.find('(')+1 : value.rfind(')')]
                low, high = map(float, content.split(','))
                parsed_space[param] = hp.uniform(param, low, high)
            elif value.startswith('randint'):
                # randint(10, 100)
                content = value[value.find('(')+1 : value.rfind(')')]
                low, high = map(int, content.split(','))
                parsed_space[param] = hp.randint(param, low, high)
            else:
                logger.warning(f"Unknown distribution for {param}: {value}")
                parsed_space[param] = value
                
        return parsed_space

    def run_tuning(self,
                   algo: str = 'random',
                   max_evals: int = 20,
                   early_stop: int = 10) -> tuple:
        """Run hyperparameter tuning"""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING HYPERPARAMETER TUNING")
        logger.info("=" * 80)
        """
        Run hyperparameter tuning
        
        Args:
            algo: Algorithm ('exhaustive', 'random', 'bayes')
            max_evals: Maximum number of evaluations
            early_stop: Stop if no improvement after N trials
            
        Returns:
            (best_result, best_params): Best results and parameters
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING HYPERPARAMETER TUNING")
        logger.info("="*80)
        logger.info(f"Algorithm: {algo}")
        logger.info(f"Max evaluations: {max_evals}")
        logger.info(f"Early stop: {early_stop}")

        config_dict = self.get_base_config()
        search_space = self.get_search_space()
        search_space = self._parse_search_space(search_space)

        fixed_config = {
            'model': self.model_name,
            'dataset': self.dataset_name,
            'data_path': str(self.data_path),
            'loss_type': 'CE',
            'train_neg_sample_args': None,
            'USER_ID_FIELD': 'session_id',
            'ITEM_ID_FIELD': 'item_id',
            'TIME_FIELD': 'timestamp',
            'SESSION_ID_FIELD': 'session_id',
            'load_col': {'inter': ['session_id', 'item_id', 'timestamp']},
        }

        exclude_keys = ['model', 'dataset', 'data_path', 'loss_type',
                        'USER_ID_FIELD', 'ITEM_ID_FIELD', 'TIME_FIELD',
                        'SESSION_ID_FIELD', 'load_col']
        tuning_config = {k: v for k, v in config_dict.items() if k not in exclude_keys}

        import tempfile
        import os
        # Cria arquivo temporário para configuração fixa
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(fixed_config, f)
            temp_config_file = f.name

        try:
            # Initialize HyperTuning com o custom_objective_function
            hp = HyperTuning(
                objective_function=custom_objective_function,  # <--- ALTERADO AQUI
                space=search_space,
                algo=algo,
                early_stop=early_stop,
                max_evals=max_evals,
                params_dict=tuning_config,
                params_file=None,
                fixed_config_file_list=[temp_config_file],
            )

            logger.info("\nStarting search...")
            best_result, best_params = hp.run()

        finally:
            if os.path.exists(temp_config_file):
                os.unlink(temp_config_file)

        logger.info("\n" + "=" * 80)
        logger.info("HYPERPARAMETER TUNING COMPLETED")

        return best_result, best_params
    
    def save_results(self, best_result: dict, best_params: dict, search_space: dict):
        """Save tuning results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_path / f'best_params_{self.model_name}_{timestamp}.json'
        
        results = {
            'model': self.model_name,
            'dataset': self.dataset_name,
            'timestamp': datetime.now().isoformat(),
            'best_params': best_params,
            'best_results': best_result,
            'search_space': search_space,
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n✅ Results saved to: {output_file}")
        
        # Print summary
        logger.info("\n🏆 BEST HYPERPARAMETERS:")
        logger.info("="*60)
        for param, value in best_params.items():
            logger.info(f"  {param}: {value}")
        
        logger.info("\n📊 BEST RESULTS:")
        logger.info("="*60)
        for metric, value in best_result.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return output_file
    
    def train_final_model(self, best_params: dict) -> dict:
        """Train final model with best hyperparameters"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
        logger.info("="*80)
        
        config = self.get_base_config()
        config.update(best_params)
        config['checkpoint_dir'] = str(self.output_path / 'final_model')
        
        logger.info("\nFinal configuration:")
        for key in ['learning_rate', 'embedding_size', 'hidden_size', 'dropout_prob']:
            if key in config:
                logger.info(f"  {key}: {config[key]}")
        
        # Train model
        result = run_recbole(
            model=self.model_name,
            dataset=self.dataset_name,
            config_dict=config
        )
        
        logger.info("\n" + "="*80)
        logger.info("FINAL MODEL TRAINING COMPLETED")
        logger.info("="*80)
        logger.info("\nTest Results:")
        for metric, value in result.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return result
    
    def save_yaml_config(self, best_params: dict):
        """Export best parameters to YAML config file"""
        config = self.get_base_config()
        config.update(best_params)
        
        # Remove non-serializable items
        config.pop('device', None)
        config['data_path'] = 'outputs/data/recbole_simple'
        config['checkpoint_dir'] = 'outputs/saved'
        
        yaml_output = Path('src') / 'configs' / 'neural' / f'{self.model_name.lower()}_tuned.yaml'
        yaml_output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_output, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"\n✅ YAML config saved to: {yaml_output}")
        logger.info("You can now use this config in your experiments!")
    
    def compare_with_baseline(self, best_params: dict):
        """Compare tuned parameters with baseline"""
        baseline = {
            'learning_rate': 0.001,
            'embedding_size': 64,
            'hidden_size': 128,
            'dropout_prob': 0.3,
        }
        
        logger.info("\n📈 COMPARISON: Baseline vs Tuned")
        logger.info("="*60)
        logger.info("\nParameter changes:")
        for param, baseline_value in baseline.items():
            tuned_value = best_params.get(param, 'N/A')
            if param in best_params:
                change = ((tuned_value - baseline_value) / baseline_value) * 100
                logger.info(f"  {param}: {baseline_value} → {tuned_value} ({change:+.1f}%)")
            else:
                logger.info(f"  {param}: {baseline_value} (unchanged)")
    
    def run(self, 
            algo: str = 'exhaustive',
            max_evals: int = 20,
            early_stop: int = 10,
            train_final: bool = True,
            save_yaml: bool = True):
        """
        Run complete tuning pipeline
        
        Args:
            algo: Tuning algorithm
            max_evals: Maximum evaluations
            early_stop: Early stopping patience
            train_final: Whether to train final model
            save_yaml: Whether to save YAML config
        """
        # Run tuning
        best_result, best_params = self.run_tuning(
            algo=algo,
            max_evals=max_evals,
            early_stop=early_stop
        )
        
        # Save results
        search_space = self.get_search_space()
        self.save_results(best_result, best_params, search_space)
        
        # Compare with baseline
        self.compare_with_baseline(best_params)
        
        # Train final model
        if train_final:
            final_result = self.train_final_model(best_params)
        
        # Save YAML config
        if save_yaml:
            self.save_yaml_config(best_params)
        
        logger.info("\n" + "="*80)
        logger.info("✅ TUNING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for RecBole models')
    parser.add_argument('--model', type=str, default='GRU4Rec',
                       help='Model name (GRU4Rec, NARM, STAMP, SASRec)')
    parser.add_argument('--dataset', type=str, default='realestate_simple',
                       help='Dataset name')
    parser.add_argument('--data-path', type=str, default='outputs/data/recbole_simple',
                       help='Path to RecBole data')
    parser.add_argument('--output-path', type=str, default='outputs/tuning',
                       help='Output directory for results')
    parser.add_argument('--algo', type=str, default='random',
                       choices=['exhaustive', 'random', 'bayes'],
                       help='Tuning algorithm')
    parser.add_argument('--max-evals', type=int, default=20,
                       help='Maximum number of evaluations')
    parser.add_argument('--early-stop', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU ID')
    parser.add_argument('--no-train-final', action='store_true',
                       help='Skip training final model')
    parser.add_argument('--no-save-yaml', action='store_true',
                       help='Skip saving YAML config')
    
    args = parser.parse_args()
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        model_name=args.model,
        dataset_name=args.dataset,
        data_path=args.data_path,
        output_path=args.output_path,
        gpu_id=args.gpu_id
    )
    
    # Run tuning
    tuner.run(
        algo=args.algo,
        max_evals=args.max_evals,
        early_stop=args.early_stop,
        train_final=not args.no_train_final,
        save_yaml=not args.no_save_yaml
    )


if __name__ == '__main__':
    main()
