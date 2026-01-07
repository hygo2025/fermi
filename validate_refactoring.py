#!/usr/bin/env python
"""
FERMI - Validation Script
==========================
Valida que a refatoração foi aplicada corretamente.
Execute antes de rodar o benchmark.
"""

import sys
from pathlib import Path
import yaml


class ValidationError(Exception):
    pass


class FermiValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed = []
        
    def log_pass(self, msg: str):
        self.passed.append(f"✅ {msg}")
        
    def log_warning(self, msg: str):
        self.warnings.append(f"⚠️  {msg}")
        
    def log_error(self, msg: str):
        self.errors.append(f"❌ {msg}")
    
    def check_project_config(self):
        """Valida config/project_config.yaml"""
        config_path = Path('config/project_config.yaml')
        
        if not config_path.exists():
            self.log_error("config/project_config.yaml não encontrado")
            return
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check eval_args
        if 'eval_args' not in config:
            self.log_error("eval_args ausente em project_config.yaml")
            return
        
        eval_args = config['eval_args']
        
        # Validate split
        if 'split' in eval_args:
            split = eval_args['split']
            if isinstance(split, dict) and split.get('LS') == 'valid_and_test':
                self.log_pass("eval_args.split configurado corretamente (LS: valid_and_test)")
            else:
                self.log_error(f"eval_args.split incorreto: {split}")
        else:
            self.log_error("eval_args.split ausente")
        
        # Validate order
        if eval_args.get('order') == 'TO':
            self.log_pass("eval_args.order = TO (Temporal Ordering ativado)")
        else:
            self.log_error(f"eval_args.order deve ser 'TO', encontrado: {eval_args.get('order')}")
        
        # Validate group_by
        if eval_args.get('group_by') == 'user':
            self.log_pass("eval_args.group_by = user (agrupa por sessão)")
        else:
            self.log_error(f"eval_args.group_by deve ser 'user', encontrado: {eval_args.get('group_by')}")
        
        # Validate mode
        if eval_args.get('mode') == 'full':
            self.log_pass("eval_args.mode = full (ranking completo)")
        else:
            self.log_warning(f"eval_args.mode recomendado: 'full', encontrado: {eval_args.get('mode')}")
        
        # Check load_col
        if 'load_col' in config and 'inter' in config['load_col']:
            cols = config['load_col']['inter']
            required = {'user_id', 'item_id', 'timestamp'}
            if set(cols) == required:
                self.log_pass("load_col.inter correto (user_id, item_id, timestamp)")
            else:
                self.log_error(f"load_col.inter deve conter {required}, encontrado: {cols}")
        else:
            self.log_error("load_col.inter ausente")
        
        # Check TIME_FIELD
        if config.get('TIME_FIELD') == 'timestamp':
            self.log_pass("TIME_FIELD = timestamp")
        else:
            self.log_error(f"TIME_FIELD deve ser 'timestamp', encontrado: {config.get('TIME_FIELD')}")
    
    def check_model_configs(self):
        """Valida configs de modelos em src/configs/"""
        config_base = Path('src/configs')
        
        if not config_base.exists():
            self.log_error("src/configs/ não encontrado")
            return
        
        # Check subdirectories
        categories = ['neural', 'baselines', 'factorization']
        for category in categories:
            cat_path = config_base / category
            if not cat_path.exists():
                self.log_warning(f"src/configs/{category}/ não encontrado")
                continue
            
            # Check YAML files
            yaml_files = list(cat_path.glob('*.yaml'))
            if not yaml_files:
                self.log_warning(f"Nenhum YAML em src/configs/{category}/")
                continue
            
            for yaml_file in yaml_files:
                with open(yaml_file, 'r') as f:
                    try:
                        config = yaml.safe_load(f)
                        
                        # Check eval_args
                        if 'eval_args' in config:
                            eval_args = config['eval_args']
                            
                            # Check group_by
                            if 'group_by' not in eval_args:
                                self.log_warning(f"{yaml_file.name}: group_by ausente em eval_args")
                            elif eval_args['group_by'] != 'user':
                                self.log_warning(f"{yaml_file.name}: group_by deveria ser 'user'")
                            
                            # Check order
                            if eval_args.get('order') != 'TO':
                                self.log_warning(f"{yaml_file.name}: order deveria ser 'TO'")
                            
                            # Check split
                            split = eval_args.get('split', {})
                            if isinstance(split, dict) and split.get('LS') != 'valid_and_test':
                                self.log_warning(f"{yaml_file.name}: split.LS deveria ser 'valid_and_test'")
                        
                    except yaml.YAMLError as e:
                        self.log_error(f"{yaml_file.name}: erro de parsing YAML - {e}")
        
        self.log_pass(f"Configs de modelo verificados em src/configs/")
    
    def check_pipeline(self):
        """Valida que o novo pipeline existe"""
        pipeline_path = Path('src/pipeline/prepare_data.py')
        
        if pipeline_path.exists():
            self.log_pass("src/pipeline/prepare_data.py encontrado")
        else:
            self.log_error("src/pipeline/prepare_data.py não encontrado")
    
    def check_benchmark_runner(self):
        """Valida que o novo orquestrador existe"""
        runner_path = Path('src/run_benchmark.py')
        
        if runner_path.exists():
            self.log_pass("src/run_benchmark.py encontrado")
        else:
            self.log_error("src/run_benchmark.py não encontrado")
    
    def check_deprecated_files(self):
        """Checa se arquivos obsoletos ainda existem"""
        deprecated_files = [
            'scripts/run_all_experiments.sh',
            'scripts/run_parallel.sh',
            'scripts/run_parallel_gpu.sh',
            'src/preprocessing/sliding_window_pipeline.py',
            'src/preprocessing/simple_data_pipeline.py',
            'src/preprocessing/recbole_converter.py',
        ]
        
        found_deprecated = []
        for file_path in deprecated_files:
            if Path(file_path).exists():
                found_deprecated.append(file_path)
        
        if found_deprecated:
            self.log_warning(f"Arquivos obsoletos ainda existem: {found_deprecated}")
            self.log_warning("Execute: bash CLEANUP_COMMANDS.sh")
        else:
            self.log_pass("Nenhum arquivo obsoleto encontrado")
    
    def run(self):
        """Executa todas as validações"""
        print("=" * 80)
        print("FERMI - VALIDATION SCRIPT")
        print("=" * 80)
        print()
        
        self.check_project_config()
        self.check_model_configs()
        self.check_pipeline()
        self.check_benchmark_runner()
        self.check_deprecated_files()
        
        # Report
        print("\n" + "=" * 80)
        print("RESULTADOS DA VALIDAÇÃO")
        print("=" * 80)
        
        if self.passed:
            print(f"\n✅ PASSOU ({len(self.passed)}):")
            for msg in self.passed:
                print(f"  {msg}")
        
        if self.warnings:
            print(f"\n⚠️  AVISOS ({len(self.warnings)}):")
            for msg in self.warnings:
                print(f"  {msg}")
        
        if self.errors:
            print(f"\n❌ ERROS ({len(self.errors)}):")
            for msg in self.errors:
                print(f"  {msg}")
        
        print("\n" + "=" * 80)
        
        if self.errors:
            print("❌ VALIDAÇÃO FALHOU - Corrija os erros antes de prosseguir")
            print("=" * 80)
            return False
        elif self.warnings:
            print("⚠️  VALIDAÇÃO PASSOU COM AVISOS - Revise as configurações")
            print("=" * 80)
            return True
        else:
            print("✅ VALIDAÇÃO PASSOU - Sistema pronto para uso!")
            print("=" * 80)
            return True


if __name__ == '__main__':
    validator = FermiValidator()
    success = validator.run()
    sys.exit(0 if success else 1)
