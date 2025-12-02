"""
Session-Based Recommendation Benchmark Runner

Quick start script to run the benchmark with session-rec framework.

Usage:
    python run_benchmark.py --config configs/default.yaml
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.prepare_dataset import SessionDataPreparator

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║          Session-Based Recommendation Benchmark for Real Estate              ║
║                  Based on: Domingues et al. (2025)                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def check_session_rec():
    """Check if session-rec is installed."""
    try:
        import session_rec
        print("✓ session-rec is installed")
        return True
    except ImportError:
        print("✗ session-rec is NOT installed")
        print("\nTo install session-rec:")
        print("  Option 1: pip install session-rec")
        print("  Option 2: pip install git+https://github.com/rn5l/session-rec.git")
        print("\nNote: session-rec may need to be installed from GitHub source")
        return False


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def step_1_prepare_data(config):
    """Step 1: Prepare dataset."""
    print("\n" + "="*80)
    print("STEP 1: DATA PREPARATION")
    print("="*80)
    
    preparator = SessionDataPreparator(
        input_path=config['data']['input_path'],
        output_path=config['data']['output_path'],
        min_session_length=config['data']['min_session_length'],
        min_item_support=config['data']['min_item_support']
    )
    
    preparator.prepare(
        start_date=config['data'].get('start_date'),
        end_date=config['data'].get('end_date')
    )
    
    print("\n✓ Data preparation completed!")


def step_2_train_models(config):
    """Step 2: Train models (placeholder)."""
    print("\n" + "="*80)
    print("STEP 2: MODEL TRAINING")
    print("="*80)
    
    print("\nThis step requires session-rec framework to be installed.")
    print("Models to train:")
    
    for model_name, model_config in config['models'].items():
        if model_config.get('enabled', False):
            print(f"  - {model_name.upper()}")
    
    print("\nImplementation coming in next steps...")
    print("See models/ directory for model implementations")


def step_3_evaluate(config):
    """Step 3: Evaluate models (placeholder)."""
    print("\n" + "="*80)
    print("STEP 3: EVALUATION")
    print("="*80)
    
    print(f"\nMetrics to compute: {', '.join(config['evaluation']['metrics'])}")
    print(f"K values: {config['evaluation']['k_values']}")
    
    print("\nImplementation coming in next steps...")
    print("See evaluation/ directory for evaluation scripts")


def main():
    parser = argparse.ArgumentParser(
        description='Run session-based recommendation benchmark'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--step',
        type=str,
        choices=['all', 'prepare', 'train', 'evaluate'],
        default='all',
        help='Which step to run'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    print("✓ Configuration loaded")
    
    # Check dependencies
    print("\nChecking dependencies...")
    has_session_rec = check_session_rec()
    
    # Run steps
    if args.step in ['all', 'prepare']:
        step_1_prepare_data(config)
    
    if args.step in ['all', 'train']:
        if has_session_rec:
            step_2_train_models(config)
        else:
            print("\n⚠ Skipping model training (session-rec not installed)")
    
    if args.step in ['all', 'evaluate']:
        if has_session_rec:
            step_3_evaluate(config)
        else:
            print("\n⚠ Skipping evaluation (session-rec not installed)")
    
    print("\n" + "="*80)
    print("BENCHMARK EXECUTION SUMMARY")
    print("="*80)
    print(f"Configuration: {args.config}")
    print(f"Step executed: {args.step}")
    
    if args.step == 'prepare' or args.step == 'all':
        print(f"\nProcessed data saved to: {config['data']['output_path']}")
        print("\nNext steps:")
        print("  1. Review processed data in", config['data']['output_path'])
        print("  2. Install session-rec if needed")
        print("  3. Run: python run_benchmark.py --step train")
    
    print("\n" + "="*80)
    print("✓ COMPLETED")
    print("="*80)


if __name__ == '__main__':
    main()
