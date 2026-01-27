#!/usr/bin/env python3
"""
Test script for KNN models (V-SKNN, STAN, V-STAN)
Validates that models can be instantiated and used with RecBole
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

from src.models import VSKNNRecommender, STANRecommender, VSTANRecommender
from src.utils.enviroment import get_config


def test_model(model_name, model_class):
    """Test a single KNN model"""
    print(f"\n{'='*80}")
    print(f"Testing {model_name}...")
    print('='*80)
    
    try:
        # Load config
        project_config = get_config()
        config_path = Path(f'src/configs/baselines/{model_name.lower()}.yaml')
        
        with open(config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        
        # Merge configs
        config_dict = {**project_config, **model_config}
        config_dict['dataset'] = project_config['dataset']
        config_dict['data_path'] = project_config['data_path']
        
        # Create RecBole config
        config = Config(
            model=model_class,
            dataset=config_dict['dataset'],
            config_dict=config_dict
        )
        
        print(f"✓ Config loaded")
        print(f"  - k: {config['k']}")
        print(f"  - sample_size: {config['sample_size']}")
        print(f"  - similarity: {config['similarity']}")
        
        # Load dataset
        print(f"\nLoading dataset...")
        dataset = create_dataset(config)
        print(f"✓ Dataset loaded")
        print(f"  - Sessions: {dataset.num(dataset.uid_field):,}")
        print(f"  - Items: {dataset.num(dataset.iid_field):,}")
        print(f"  - Interactions: {len(dataset):,}")
        
        # Create dataloaders
        print(f"\nCreating dataloaders...")
        train_data, valid_data, test_data = data_preparation(config, dataset)
        print(f"✓ Dataloaders created")
        print(f"  - Train batches: {len(train_data)}")
        print(f"  - Valid batches: {len(valid_data)}")
        print(f"  - Test batches: {len(test_data)}")
        
        # Initialize model
        print(f"\nInitializing model...")
        model = model_class(config, dataset).to(config['device'])
        print(f"✓ Model initialized")
        print(f"  - Device: {config['device']}")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test training loop (build mappings)
        print(f"\nBuilding session mappings...")
        model.train()
        for batch_idx, batch in enumerate(train_data):
            batch = batch.to(config['device'])
            loss = model.calculate_loss(batch)
            if batch_idx % 100 == 0:
                print(f"  - Batch {batch_idx}/{len(train_data)}", end='\r')
        
        print(f"\n✓ Session mappings built")
        print(f"  - Sessions: {len(model.session_item_map):,}")
        print(f"  - Items in map: {len(model.item_session_map):,}")
        
        # Test prediction
        print(f"\nTesting prediction...")
        model.eval()
        with torch.no_grad():
            test_batch = next(iter(test_data)).to(config['device'])
            scores = model.full_sort_predict(test_batch)
            print(f"✓ Prediction successful")
            print(f"  - Output shape: {scores.shape}")
            print(f"  - Score range: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"  - Non-zero scores: {(scores > 0).sum().item():,}")
        
        print(f"\n✅ {model_name} PASSED all tests!")
        return True
        
    except Exception as e:
        print(f"\n❌ {model_name} FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all KNN models"""
    print("\n" + "="*80)
    print("KNN Models Test Suite")
    print("="*80)
    
    models = [
        ('VSKNN', VSKNNRecommender),
        ('STAN', STANRecommender),
        ('VSTAN', VSTANRecommender),
    ]
    
    results = {}
    for model_name, model_class in models:
        results[model_name] = test_model(model_name, model_class)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for model_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{model_name:15s} {status}")
    
    all_passed = all(results.values())
    
    print("="*80)
    if all_passed:
        print("🎉 All models passed!")
        return 0
    else:
        print("⚠️  Some models failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
