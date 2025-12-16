import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

# Monkey-patch torch.load para PyTorch 2.6+ compatibilidade
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec, NARM, STAMP, SASRec, FPMC, FOSSIL
from recbole.trainer import Trainer
from recbole.utils import init_seed

from src.metrics import SessionBasedMetrics
from src.utils.gpu_cooling import inject_cooling_callback
from src.models import RandomRecommender, POPRecommender, RPOPRecommender, SPOPRecommender

class ExperimentRunner:
    """Executa experimentos de recomendação em múltiplos slices"""
    
    def __init__(self, 
                 data_path: str = 'recbole_data',
                 output_path: str = 'results',
                 config_path: str = 'src/configs',
                 models: List[str] = None,
                 slices: List[int] = None,
                 enable_gpu_cooling: bool = True,
                 cool_every_n_epochs: int = 5,
                 cool_duration_seconds: int = 60,
                 max_temp_celsius: int = 80,
                 shared_timestamp: str = None,
                 save_checkpoints: bool = False):
        """
        Args:
            data_path: Diretório com dados RecBole
            output_path: Diretório para salvar resultados
            config_path: Diretório com configs YAML dos modelos
            models: Lista de modelos para executar (None = todos)
            slices: Lista de slices para processar (None = todos)
            enable_gpu_cooling: Ativar pausas para resfriamento
            cool_every_n_epochs: Pausar a cada N epochs
            cool_duration_seconds: Duração da pausa em segundos
            max_temp_celsius: Temperatura máxima antes de forçar pausa
            shared_timestamp: Timestamp compartilhado (para run_all)
            save_checkpoints: Se True, salva checkpoints (padrão: False para economizar espaço)
        """
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.save_checkpoints = save_checkpoints
        
        # Create timestamped output directory
        # Use shared timestamp if provided, otherwise create new
        if shared_timestamp:
            timestamp = shared_timestamp
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.output_path = Path(output_path) / timestamp
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now()
        
        # GPU Cooling settings
        self.enable_gpu_cooling = enable_gpu_cooling
        self.cool_every_n_epochs = cool_every_n_epochs
        self.cool_duration_seconds = cool_duration_seconds
        self.max_temp_celsius = max_temp_celsius
        
        # Initialize override parameters
        self.override_epochs = None
        self.override_batch_size = None
        
        # Modelos disponíveis
        self.available_models = {
            'GRU4Rec': GRU4Rec,
            'NARM': NARM,
            'STAMP': STAMP,
            'SASRec': SASRec,
            'FPMC': FPMC,
            'FOSSIL': FOSSIL,
            'Random': RandomRecommender,
            'POP': POPRecommender,
            'RPOP': RPOPRecommender,
            'SPOP': SPOPRecommender,
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
        # Try neural models first, then baselines, then factorization
        config_file = self.config_path / 'neural' / f'{model_name.lower()}.yaml'
        
        if not config_file.exists():
            config_file = self.config_path / 'baselines' / f'{model_name.lower()}.yaml'
        
        if not config_file.exists():
            config_file = self.config_path / 'factorization' / f'{model_name.lower()}.yaml'
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found for model: {model_name}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override dataset and data_path for current slice
        config['dataset'] = dataset_name
        config['data_path'] = str(self.data_path)
        
        # Controle de salvamento de checkpoints
        if not self.save_checkpoints:
            # Desabilitar salvamento para economizar espaço
            config['checkpoint_dir'] = None
        
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
        
        # Ensure eval_step is set
        if 'eval_step' not in config_dict:
            config_dict['eval_step'] = 1
        
        # For custom models, don't let RecBole try to import them
        if model_name in ['Random', 'POP', 'RPOP', 'SPOP']:
            config = Config(model='GRU4Rec', config_dict=config_dict)  # Use GRU4Rec as template
            config['model'] = model_name  # Override model name
        else:
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
        
        # Create loss history tracker
        train_loss_history = []
        
        # Hook into trainer's _train_epoch to capture losses
        original_train_epoch = trainer._train_epoch
        def _train_epoch_with_tracking(train_data, epoch_idx, loss_func=None, show_progress=False):
            total_loss = original_train_epoch(train_data, epoch_idx, loss_func, show_progress)
            train_loss_history.append(float(total_loss))
            return total_loss
        trainer._train_epoch = _train_epoch_with_tracking
        
        # Inject GPU cooling callback if enabled
        if self.enable_gpu_cooling:
            self.logger.info(f"  GPU Cooling enabled: pause every {self.cool_every_n_epochs} epochs for {self.cool_duration_seconds}s")
            self.logger.info(f"  Max temperature threshold: {self.max_temp_celsius}°C")
            inject_cooling_callback(
                trainer,
                cool_every_n_epochs=self.cool_every_n_epochs,
                cool_duration_seconds=self.cool_duration_seconds,
                max_temp_celsius=self.max_temp_celsius
            )
        
        # Train and track losses
        self.logger.info(f"  Training {model_name}...")
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, show_progress=True
        )
        
        # Test
        self.logger.info(f"  Testing {model_name}...")
        test_result = trainer.evaluate(test_data, show_progress=True)
        
        # Save loss curves
        loss_info = {
            'train_losses': train_loss_history
        }
        
        # Save loss data to file
        loss_file = self.output_path / 'losses' / f'{model_name}_slice{slice_id}_loss.json'
        loss_file.parent.mkdir(exist_ok=True)
        
        import json
        with open(loss_file, 'w') as f:
            json.dump(loss_info, f, indent=2)
        
        self.logger.info(f"  Loss history saved: {len(train_loss_history)} epochs")
        
        # Format results
        results = {
            'model': model_name,
            'slice': slice_id,
            'dataset': dataset_name,
            'best_valid_score': best_valid_score,
            'total_epochs': len(train_loss_history),
            **test_result
        }
        
        self.logger.info(f"  Results: {test_result}")
        
        return results
    
    def run_all_experiments(self):
        """Executa todos os experimentos e gera resultados agregados"""
        self.logger.info("="*80)
        self.logger.info("Starting Session-Based Recommendation Experiments")
        self.logger.info("="*80)
        self.logger.info(f"Output directory: {self.output_path}")
        self.logger.info(f"Models: {self.models}")
        self.logger.info(f"Slices: {self.slices}")
        self.logger.info("")
        
        all_results = []
        
        for model_name in self.models:
            for slice_id in self.slices:
                try:
                    result = self.run_single_experiment(model_name, slice_id)
                    all_results.append(result)
                    
                    # Save result immediately (incremental save with duplicate prevention)
                    df_new = pd.DataFrame([result])
                    output_file = self.output_path / 'raw_results.csv'
                    
                    if output_file.exists():
                        df_existing = pd.read_csv(output_file)
                        # Remove any existing entry for this model+slice combo
                        df_existing = df_existing[~((df_existing['model'] == model_name) & 
                                                    (df_existing['slice'] == slice_id))]
                        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                        df_combined.to_csv(output_file, index=False)
                    else:
                        df_new.to_csv(output_file, index=False)
                    
                    self.logger.info(f"Progress: {len(all_results)}/{len(self.models) * len(self.slices)} completed")
                    
                except Exception as e:
                    self.logger.error(f"Error running {model_name} on slice {slice_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Aggregate and generate visualizations
        if hasattr(self, 'skip_aggregation') and self.skip_aggregation:
            self.logger.info("\n" + "="*80)
            self.logger.info("WARNING: Aggregation skipped (--no-aggregate flag)")
            self.logger.info(f"Raw results saved to: {self.output_path}")
            self.logger.info("="*80)
        else:
            self.logger.info("\n" + "="*80)
            self.logger.info("Aggregating results and generating visualizations...")
            self.logger.info("="*80)
            
            from src.aggregate_results import ResultsAggregator
            
            aggregator = ResultsAggregator(
                input_path=str(self.output_path),
                output_dir=str(self.output_path / 'aggregated.csv'),
                create_timestamped=False  # We already have a timestamped dir
            )
            aggregator.run()
            
            self.logger.info("\n" + "="*80)
            self.logger.info("All experiments completed!")
            self.logger.info(f"Results saved to: {self.output_path}")
            self.logger.info("="*80)
        
        return all_results
    
    def save_results(self, results: List[Dict]):
        """Salva resultados em CSV (append mode para runs paralelos)"""
        df = pd.DataFrame(results)
        output_file = self.output_path / 'raw_results.csv'
        
        # Append if file exists, otherwise create new
        if output_file.exists():
            existing_df = pd.read_csv(output_file)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(output_file, index=False)
        self.logger.info(f"Results saved to {output_file} ({len(df)} total)")


def main():
    parser = argparse.ArgumentParser(description='Run session-based recommendation experiments')
    parser.add_argument('--data-path', type=str, default='outputs/data/recbole',
                       help='Path to RecBole data')
    parser.add_argument('--output-path', type=str, default='outputs/results',
                       help='Path to save results')
    parser.add_argument('--config-path', type=str, default='src/configs',
                       help='Path to model config files')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Models to run (default: all)')
    parser.add_argument('--slices', type=int, nargs='+', default=None,
                       help='Slices to process (default: all)')
    parser.add_argument('--all-slices', action='store_true',
                       help='Run on all slices')
    
    # GPU Cooling arguments
    parser.add_argument('--enable-gpu-cooling', action='store_true', default=True,
                       help='Enable GPU cooling breaks (default: True)')
    parser.add_argument('--no-gpu-cooling', dest='enable_gpu_cooling', action='store_false',
                       help='Disable GPU cooling breaks')
    parser.add_argument('--cool-every', type=int, default=15,
                       help='Cool down every N epochs (default: 15)')
    parser.add_argument('--cool-duration', type=int, default=20,
                       help='Cooling duration in seconds (default: 20)')
    parser.add_argument('--max-temp', type=int, default=80,
                       help='Max GPU temperature before forced cooling (default: 80°C)')
    parser.add_argument('--no-aggregate', action='store_true',
                       help='Skip automatic aggregation (for parallel runs)')
    parser.add_argument('--shared-timestamp', type=str, default=None,
                       help='Shared timestamp for parallel execution')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs (for testing)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size (for testing)')
    parser.add_argument('--save-checkpoints', action='store_true',
                       help='Save model checkpoints (default: False para economizar espaço)')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        data_path=args.data_path,
        output_path=args.output_path,
        config_path=args.config_path,
        models=args.models,
        slices=args.slices,
        enable_gpu_cooling=args.enable_gpu_cooling,
        cool_every_n_epochs=args.cool_every,
        cool_duration_seconds=args.cool_duration,
        max_temp_celsius=args.max_temp,
        shared_timestamp=args.shared_timestamp,
        save_checkpoints=args.save_checkpoints
    )
    
    # Set override parameters if provided
    if args.epochs:
        runner.override_epochs = args.epochs
    if args.batch_size:
        runner.override_batch_size = args.batch_size


    # Skip aggregation if requested (for parallel runs)
    if args.no_aggregate:
        runner.skip_aggregation = True

    runner.run_all_experiments()


if __name__ == '__main__':
    main()
