"""
Run Session-Rec Benchmark

Wrapper to run session-rec framework with our data.
"""

import sys
import os
import argparse
import yaml
from pathlib import Path

# Add session-rec to path
PROJECT_DIR = Path(__file__).parent.parent
SESSION_REC_DIR = PROJECT_DIR / 'session-rec-lib'
BENCHMARK_DIR = PROJECT_DIR / 'src'
sys.path.insert(0, str(SESSION_REC_DIR))

# Import session-rec runner at module level
from run_config import main as run_session_rec_main

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 Session-Rec Benchmark Runner                                 ║
║          Real Estate Session-Based Recommendation Benchmark                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def run_benchmark(config_file):
    """
    Run session-rec with our configuration.
    
    Args:
        config_file: Path to YAML config file
    """
    print(f"Loading configuration: {config_file}")
    
    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"  Type: {config['type']}")
    print(f"  Key: {config['key']}")
    print(f"  Data: {config['data']['name']}")
    print(f"  Folder: {config['data']['folder']}")
    print(f"  Prefix: {config['data']['prefix']}")
    
    # Count algorithms
    n_algos = len(config['algorithms'])
    print(f"  Algorithms: {n_algos}")
    for algo in config['algorithms']:
        print(f"    - {algo['key']}: {algo['class']}")
    
    print("\n" + "="*80)
    print("STARTING BENCHMARK")
    print("="*80)
    
    # Import session-rec runner
    # Convert to absolute path
    config_path = Path(config_file).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        # Change to session-rec directory for relative paths
        original_dir = os.getcwd()
        os.chdir(str(SESSION_REC_DIR))
        
        # Run with our config
        run_session_rec_main(conf=str(config_path))
        
        # Return to original directory
        os.chdir(original_dir)
        
    except Exception as e:
        os.chdir(original_dir)  # Ensure we return to original dir
        print(f"\n❌ Error running session-rec: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "="*80)
        print("TROUBLESHOOTING TIPS")
        print("="*80)
        print("1. Make sure data is in the correct format")
        print("2. Check that all paths in config are correct")
        print("3. Verify session-rec dependencies are installed")
        print("\nFor manual execution:")
        print(f"  cd {SESSION_REC_DIR}")
        print(f"  python run_config.py {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run session-rec benchmark with our data'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/session_rec_config.yml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--list-algorithms',
        action='store_true',
        help='List available algorithms'
    )
    
    args = parser.parse_args()
    
    if args.list_algorithms:
        print("Available Algorithms in session-rec:")
        print("\nBaselines:")
        algos_dir = SESSION_REC_DIR / 'algorithms' / 'baselines'
        for f in sorted(algos_dir.glob('*.py')):
            if f.stem != '__init__':
                print(f"  - baselines.{f.stem}")
        
        print("\nKNN-based:")
        algos_dir = SESSION_REC_DIR / 'algorithms' / 'knn'
        for f in sorted(algos_dir.glob('*.py')):
            if f.stem != '__init__':
                print(f"  - knn.{f.stem}")
        
        print("\nDeep Learning:")
        for subdir in ['gru4rec', 'narm', 'STAMP', 'nsar']:
            if (SESSION_REC_DIR / 'algorithms' / subdir).exists():
                print(f"  - {subdir}")
        
        return
    
    # Run benchmark
    run_benchmark(args.config)


if __name__ == '__main__':
    main()
