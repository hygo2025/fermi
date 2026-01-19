import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import yaml

import pandas as pd
import torch

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

from src.models import RandomRecommender, POPRecommender, RPOPRecommender, SPOPRecommender
from src.utils import log
from src.utils.enviroment import get_config

# Custom models need special handling
CUSTOM_MODELS = {
    'Random': RandomRecommender,
    'POP': POPRecommender,
    'RPOP': RPOPRecommender,
    'SPOP': SPOPRecommender,
}

MODEL_CONFIG_DIRS = ['neural', 'factorization', 'baselines']

class BenchmarkRunner:
    def __init__(self, output_dir: Optional[str] = None):
        self.project_config = get_config()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            base_dir = Path(self.project_config['output']['results_dir'])
            self.output_dir = base_dir / self.timestamp

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
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
            raise FileNotFoundError(f"Config não encontrado para modelo: {model_name}")

        # Merge com project config (projeto override modelo)
        config_dict = {**self.project_config, **model_config}
        config_dict['dataset'] = dataset_name
        config_dict['data_path'] = self.project_config['data_path']
        
        model_output_dir = self.output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        config_dict['checkpoint_dir'] = str(model_output_dir / 'checkpoints')
        
        # Wandb: define nome do run via variável de ambiente (RecBole não aceita via config)
        if config_dict.get('log_wandb', False):
            import os
            os.environ['WANDB_NAME'] = f"{model_name}_{self.timestamp}"
        
        return config_dict

    def run_single_model(self, model_name: str, dataset_name: str) -> dict:
        """Executa um único modelo"""
        log(f"{'=' * 80}")
        log(f"Executando: {model_name} | Dataset: {dataset_name}")
        log(f"{'=' * 80}")

        try:
            # Load config
            config_dict = self._get_model_config(model_name, dataset_name)
            
            # Configura logging específico por modelo (para tensorboard usar nome correto)
            model_log_file = self.output_dir / 'logs' / f'{model_name}.log'
            model_log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Remove handlers antigos e adiciona handler específico do modelo
            root_logger = logging.getLogger()
            # Guarda handlers originais
            original_handlers = root_logger.handlers.copy()
            # Limpa handlers temporariamente
            root_logger.handlers.clear()
            # Adiciona handler do modelo (tensorboard vai usar este nome)
            model_handler = logging.FileHandler(model_log_file)
            model_handler.setFormatter(logging.Formatter('%(message)s'))
            root_logger.addHandler(model_handler)
            root_logger.addHandler(logging.StreamHandler(sys.stdout))

            # Para custom models, usa template e override
            if model_name in CUSTOM_MODELS:
                config = Config(model='GRU4Rec', config_dict=config_dict)
                config['model'] = model_name
            else:
                config = Config(model=model_name, config_dict=config_dict)

            # Init seed
            init_seed(config['seed'], config['reproducibility'])

            # Load dataset
            log("Carregando dataset...")
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)

            # Create model
            log("Instanciando modelo...")
            if model_name in CUSTOM_MODELS:
                model_class = CUSTOM_MODELS[model_name]
                model = model_class(config, train_data.dataset).to(config['device'])
            else:
                # RecBole models are loaded automatically via Config
                from recbole.utils import get_model
                model = get_model(model_name)(config, train_data.dataset).to(config['device'])

            # Create trainer
            trainer = Trainer(config, model)

            # Train
            log("Iniciando treinamento...")
            best_valid_score, best_valid_result = trainer.fit(
                train_data, valid_data, show_progress=True
            )

            # Test
            log("Avaliando no conjunto de teste...")
            test_result = trainer.evaluate(test_data, show_progress=True)

            # Format results
            results = {
                'model': model_name,
                'dataset': dataset_name,
                'best_valid_score': best_valid_score,
                'timestamp': self.timestamp,
                **test_result
            }

            log(f" Resultados: {test_result}")

            # Restaura handlers originais do logger
            root_logger.handlers.clear()
            root_logger.handlers.extend(original_handlers)

            return results

        except Exception as e:
            log(f" ERRO ao executar {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Restaura handlers em caso de erro também
            root_logger = logging.getLogger()
            if 'original_handlers' in locals():
                root_logger.handlers.clear()
                root_logger.handlers.extend(original_handlers)
            
            return {
                'model': model_name,
                'dataset': dataset_name,
                'error': str(e),
                'timestamp': self.timestamp
            }

    def run(self, models: List[str], dataset: str):
        """Executa benchmark para lista de modelos"""
        log(f"{'=' * 80}")
        log(f"Dataset: {dataset}")
        log(f"Modelos: {models}")
        log(f"Output base: {self.output_dir}")
        log(f"{'=' * 80}")

        # Se rodar múltiplos modelos, cria subdir por modelo
        # Se rodar 1 modelo, usa dir direto
        if len(models) > 1:
            base_output = self.output_dir
        else:
            # Modelo único: renomeia output_dir para incluir nome do modelo
            base_dir = Path(self.project_config['output']['results_dir'])
            self.output_dir = base_dir / f"{self.timestamp}_{models[0]}"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            base_output = self.output_dir.parent

        results_file = base_output / f'results_{self.timestamp}.csv'
        all_results = []

        for i, model_name in enumerate(models, 1):
            log(f"[{i}/{len(models)}] Processando: {model_name}")

            result = self.run_single_model(model_name, dataset)
            all_results.append(result)

            # Save incremental
            df = pd.DataFrame(all_results)
            df.to_csv(results_file, index=False)
            log(f"Resultado salvo: {results_file}")

        log(f"{'=' * 80}")
        log(f"BENCHMARK COMPLETO!")
        log(f"{'=' * 80}")
        log(f"Total de modelos executados: {len(all_results)}")
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
        log(f"ERRO: Config não encontrado para modelo '{args.model}'")
        log(f"Procurado em: src/configs/{{neural,baselines,factorization}}/{args.model.lower()}.yaml")
        sys.exit(1)

    # Load dataset from config if not specified
    if not args.dataset:
        dataset = get_config('dataset')
    else:
        dataset = args.dataset

    # Run benchmark for single model
    runner = BenchmarkRunner(args.output)
    runner.run([args.model], dataset)


if __name__ == '__main__':
    main()
