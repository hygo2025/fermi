import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml

import pandas as pd
import torch


import scipy.sparse as sp

if not hasattr(sp.dok_matrix, "_update"):
    def _update(self, data_dict):
        for k, v in data_dict.items():
            self[k] = v


    sp.dok_matrix._update = _update
# Configure PyTorch CUDA memory allocator to reduce fragmentation
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Monkey-patch torch.load (PyTorch 2.6+ compatibility)
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed

from src.utils import log
from src.utils.enviroment import get_config

CUSTOM_MODELS = {}

MODEL_CONFIG_DIRS = ['neural', 'factorization', 'baselines']

class BenchmarkRunner:
    def __init__(self, output_dir: Optional[str] = None):
        self.project_config = get_config()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            base_dir = Path(self.project_config['output']['results_dir'])
            self.output_dir = base_dir / self.timestamp

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

    def _setup_logging(self):
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'benchmark_{self.timestamp}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def _get_model_config(self, model_name: str, dataset_name: str) -> dict:
        """Carrega config do modelo e merge com config do projeto"""
        config_base = Path('src/configs')

        for category in ['neural', 'baselines', 'factorization']:
            config_file = config_base / category / f'{model_name.lower()}.yaml'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    model_config = yaml.safe_load(f)
                break
        else:
            raise FileNotFoundError(f"Config not found for model: {model_name}")

        config_dict = {**self.project_config, **model_config}
        config_dict['dataset'] = dataset_name
        config_dict['data_path'] = self.project_config['data_path']

        # Use shared checkpoint_dir from project_config.yaml for dataset cache
        # Model checkpoints will be saved with unique names
        config_dict['show_progress'] = True
        
        if config_dict.get('log_wandb', False):
            import os
            os.environ['WANDB_NAME'] = f"{model_name}_{self.timestamp}"
            if 'wandb_group' in config_dict and config_dict['wandb_group']:
                os.environ['WANDB_RUN_GROUP'] = config_dict['wandb_group']
        
        return config_dict

    def run_single_model(self, model_name: str, dataset_name: str, run_evaluate: bool = False) -> dict:
        log(f"{'=' * 80}")
        log(f"Running: {model_name} | Dataset: {dataset_name}")
        log(f"{'=' * 80}")

        try:
            config_dict = self._get_model_config(model_name, dataset_name)

            if model_name in CUSTOM_MODELS:
                config = Config(model='GRU4Rec', config_dict=config_dict)
                config['model'] = model_name
            else:
                config = Config(model=model_name, config_dict=config_dict)

            init_seed(config['seed'], config['reproducibility'])

            log("Loading dataset...")
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)

            log("Initializing model...")
            if model_name in CUSTOM_MODELS:
                model_class = CUSTOM_MODELS[model_name]
                model = model_class(config, train_data.dataset).to(config['device'])
            else:
                from recbole.utils import get_model
                model = get_model(model_name)(config, train_data.dataset).to(config['device'])
            log("Initializing trainer...")
            trainer = Trainer(config, model)

            log("Training...")
            best_valid_score, best_valid_result = trainer.fit(
                train_data, valid_data, show_progress=True
            )
            
            # Clear GPU cache after training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log(f"GPU memory after training: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

            test_result = {}
            if run_evaluate:
                log("Evaluating...")
                # Clear GPU cache before evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    log(f"GPU memory before eval: {torch.cuda.memory_allocated(0)/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

                test_result = trainer.evaluate(test_data, show_progress=True)

                # Clear GPU cache after evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Log test metrics to W&B
                if config['log_wandb']:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({f'test_{k}': v for k, v in test_result.items()})

            # Format results
            results = {
                'model': model_name,
                'dataset': dataset_name,
                'best_valid_score': best_valid_score,
                'timestamp': self.timestamp,
                **test_result
            }

            log(f" Resultados: {test_result}")


            return results

        except Exception as e:
            log(f" ERRO ao executar {model_name}: {e}")
            import traceback
            traceback.print_exc()

            return {
                'model': model_name,
                'dataset': dataset_name,
                'error': str(e),
                'timestamp': self.timestamp
            }

    def run_single(self, model_name: str, dataset: str, run_evaluate: bool = False):
        """Executa benchmark para um único modelo"""
        log(f"{'=' * 80}")
        log(f"Dataset: {dataset}")
        log(f"Modelo: {model_name}")
        log(f"Output base: {self.output_dir}")
        log(f"{'=' * 80}")

        results_file = self.output_dir / f'results_{self.timestamp}.csv'

        result = self.run_single_model(model_name, dataset, run_evaluate=run_evaluate)
        df = pd.DataFrame([result])
        df.to_csv(results_file, index=False)
        log(f"Resultado salvo: {results_file}")

        log(f"{'=' * 80}")
        log(f"BENCHMARK COMPLETO!")
        log(f"{'=' * 80}")
        log(f"Resultados salvos em: {results_file}")
        log(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description='Fermi Benchmark Runner - Executa um modelo específico',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Nome do modelo a executar (ex: GRU4Rec, SASRec, etc.)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Nome do dataset (default: usa config do projeto)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Diretório de output customizado'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Executa avaliação após o treino (default: false)'
    )

    args = parser.parse_args()

    # Validate model config exists
    config_base = Path('src/configs')
    model_found = False
    for category in MODEL_CONFIG_DIRS:
        config_file = config_base / category / f'{args.model.lower()}.yaml'
        if config_file.exists():
            model_found = True
            break

    if not model_found:
        log(f"ERROR: Config not found for model '{args.model}'")
        log(f"Searched in: src/configs/{{neural,baselines,factorization}}/{args.model.lower()}.yaml")
        sys.exit(1)

    # Load dataset from config if not specified
    if not args.dataset:
        dataset = get_config('dataset')
    else:
        dataset = args.dataset

    # Run benchmark for single model
    runner = BenchmarkRunner(args.output)
    runner.run_single(args.model, dataset, run_evaluate=args.evaluate)


if __name__ == '__main__':
    main()
