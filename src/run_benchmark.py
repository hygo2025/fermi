import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import yaml

# Monkey-patch torch.load (PyTorch 2.6+ compatibility)
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec, NARM, STAMP, SASRec, FPMC, FOSSIL, BERT4Rec, SRGNN, Caser, GCSAN
from recbole.trainer import Trainer
from recbole.utils import init_seed

from src.models import RandomRecommender, POPRecommender, RPOPRecommender, SPOPRecommender
from src.utils import log
from src.utils.enviroment import get_config

MODEL_REGISTRY = {
    'GRU4Rec': GRU4Rec,
    'NARM': NARM,
    'STAMP': STAMP,
    'SASRec': SASRec,
    'BERT4Rec': BERT4Rec,
    'SRGNN': SRGNN,
    'Caser': Caser,
    'GCSAN': GCSAN,
    'FPMC': FPMC,
    'FOSSIL': FOSSIL,
    'Random': RandomRecommender,
    'POP': POPRecommender,
    'RPOP': RPOPRecommender,
    'SPOP': SPOPRecommender,
}

MODEL_GROUPS = {
    'neurais': ['GRU4Rec', 'NARM', 'STAMP', 'SASRec', 'BERT4Rec', 'SRGNN', 'Caser', 'GCSAN'],
    'baselines': ['Random', 'POP', 'RPOP', 'SPOP'],
    'factorization': ['FPMC', 'FOSSIL'],
    'all': list(MODEL_REGISTRY.keys())
}


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
        # Busca config específica do modelo
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

        return config_dict

    def run_single_model(self, model_name: str, dataset_name: str) -> dict:
        """Executa um único modelo"""
        log(f"{'=' * 80}")
        log(f"Executando: {model_name} | Dataset: {dataset_name}")
        log(f"{'=' * 80}")

        try:
            # Load config
            config_dict = self._get_model_config(model_name, dataset_name)

            # Para custom models, usa template e override
            if model_name in ['Random', 'POP', 'RPOP', 'SPOP']:
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
            model_class = MODEL_REGISTRY[model_name]
            model = model_class(config, train_data.dataset).to(config['device'])

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

    def run(self, models: List[str], dataset: str):
        """Executa benchmark para lista de modelos"""
        log(f"{'=' * 80}")
        log(f"Dataset: {dataset}")
        log(f"Modelos: {models}")
        log(f"Output: {self.output_dir}")
        log(f"{'=' * 80}")

        results_file = self.output_dir / 'results.csv'
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


def parse_models(model_input: List[str]) -> List[str]:
    """Parse model input (pode ser grupo ou lista de modelos)"""
    models = []
    for item in model_input:
        if item in MODEL_GROUPS:
            models.extend(MODEL_GROUPS[item])
        elif item in MODEL_REGISTRY:
            models.append(item)
        else:
            raise ValueError(f"Modelo/grupo desconhecido: {item}")

    # Remove duplicatas mantendo ordem
    return list(dict.fromkeys(models))


def main():
    parser = argparse.ArgumentParser(
        description='Fermi Benchmark Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='Lista de modelos ou grupos (neurais, baselines, factorization, all)'
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

    # Parse models
    try:
        models = parse_models(args.models)
    except ValueError as e:
        log(f"ERRO: {e}")
        log(f"Modelos disponíveis: {list(MODEL_REGISTRY.keys())}")
        log(f"Grupos disponíveis: {list(MODEL_GROUPS.keys())}")
        sys.exit(1)

    # Load dataset from config if not specified
    if not args.dataset:
        dataset = get_config('dataset')
    else:
        dataset = args.dataset

    # Run benchmark
    runner = BenchmarkRunner(args.output)
    runner.run(models, dataset)


if __name__ == '__main__':
    main()
