import argparse
import yaml
from pathlib import Path
from recbole.quick_start import run_recbole


def run_benchmark(config_file):
    """
    Run RecBole benchmark with given config
    
    Args:
        config_file: Path to YAML config file
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 RecBole Benchmark Runner                                     ║
║          Real Estate Session-Based Recommendation Benchmark                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Loading configuration: {config_file}
""")
    
    # Load config to show details
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"  Model: {config.get('model', 'N/A')}")
    print(f"  Dataset: {config.get('dataset', 'N/A')}")
    print(f"  Epochs: {config.get('epochs', 'N/A')}")
    
    print("\n" + "="*80)
    print("STARTING BENCHMARK")
    print("="*80 + "\n")
    
    # Run RecBole
    try:
        result = run_recbole(
            model=config.get('model'),
            dataset=config.get('dataset'),
            config_file_list=[str(config_path)]
        )
        
        print("\n" + "="*80)
        print("BENCHMARK COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nResults: {result}")
        
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR RUNNING BENCHMARK")
        print("="*80)
        print(f"\nError: {e}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RecBole benchmark')
    parser.add_argument('--config', '-c', required=True, help='Path to config file')
    
    args = parser.parse_args()
    run_benchmark(args.config)
