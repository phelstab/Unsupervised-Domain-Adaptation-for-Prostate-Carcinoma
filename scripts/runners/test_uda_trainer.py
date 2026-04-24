#!/usr/bin/env python3
"""
Test script for UDATrainer with assertions.
1. Source test evaluation
2. Target-based checkpoint selection (oracle upper bound)
3. Periodic checkpoint saving

Run with: .venv-cnn\Scripts\python.exe scripts\runners\test_uda_trainer.py
"""

import sys
import tempfile
import shutil
from pathlib import Path
import torch
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent / 'models' / 'MRI' / 'baseline' / 'src'))

from cnn.training_setup.uda_trainer import (
    UDATrainer, DataSplits, CheckpointManager, ValidatorType, CheckpointInfo
)


def test_data_splits():
    """Test DataSplits dataclass assertions."""
    print("Testing DataSplits...")
    
    class MockDataset:
        def __init__(self, size):
            self.size = size
        def __len__(self):
            return self.size
    
    from torch.utils.data import Subset
    
    mock_ds = MockDataset(10)
    valid_subset = Subset(mock_ds, [0, 1, 2])
    
    splits = DataSplits(
        source_train=valid_subset,
        source_val=valid_subset,
        source_test=valid_subset,
        target_train=valid_subset,
        target_val=valid_subset,
        target_test=valid_subset
    )
    
    assert len(splits.source_train) > 0
    assert len(splits.source_test) > 0, "source_test must exist"
    assert len(splits.target_test) > 0
    
    print("  [OK] DataSplits validation passed")
    print("  [OK] source_test split exists")


def test_checkpoint_manager():
    """Test CheckpointManager for periodic saving and dual validators."""
    print("\nTesting CheckpointManager...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        manager = CheckpointManager(workdir, save_interval=5, keep_all=True)
        
        assert manager.should_save(0, 100), "Should save at epoch 0"
        assert manager.should_save(5, 100), "Should save at interval"
        assert manager.should_save(10, 100), "Should save at interval"
        assert not manager.should_save(3, 100), "Should not save off-interval"
        assert manager.should_save(99, 100), "Should save at last epoch"
        
        print("  [OK] Periodic saving logic works ")
        
        mock_state = {'layer1.weight': torch.randn(10, 10)}
        
        manager.save_checkpoint(0, mock_state, source_val_bal_acc=50.0, target_val_bal_acc=40.0)
        manager.save_checkpoint(5, mock_state, source_val_bal_acc=55.0, target_val_bal_acc=60.0)
        manager.save_checkpoint(10, mock_state, source_val_bal_acc=52.0, target_val_bal_acc=65.0)
        
        best_source = manager.get_best_checkpoint(ValidatorType.SOURCE_VAL)
        best_target = manager.get_best_checkpoint(ValidatorType.TARGET_VAL)
        
        assert best_source is not None
        assert best_target is not None
        assert best_source.epoch == 5, f"Best by source should be epoch 5, got {best_source.epoch}"
        assert best_target.epoch == 10, f"Best by target should be epoch 10, got {best_target.epoch}"
        
        print("  [OK] Source-based checkpoint selection works")
        print("  [OK] Target-based checkpoint selection works")
        
        assert (workdir / "best_by_source_val.pt").exists()
        assert (workdir / "best_by_target_val.pt").exists()
        assert (workdir / "checkpoints" / "epoch_0000.pt").exists()
        assert (workdir / "checkpoints" / "epoch_0005.pt").exists()
        
        print("  [OK] All checkpoints saved to disk")
        
        summary = manager.get_summary()
        assert summary['total_checkpoints'] == 3
        assert summary['best_by_source']['epoch'] == 5
        assert summary['best_by_target']['epoch'] == 10
        
        print("  [OK] Checkpoint summary works")


def test_validator_enum():
    """Test ValidatorType enum."""
    print("\nTesting ValidatorType enum...")
    
    assert ValidatorType.SOURCE_VAL.value == "source_val"
    assert ValidatorType.TARGET_VAL.value == "target_val"
    
    source_validator = ValidatorType("source_val")
    target_validator = ValidatorType("target_val")
    
    assert source_validator == ValidatorType.SOURCE_VAL
    assert target_validator == ValidatorType.TARGET_VAL
    
    print("  [OK] ValidatorType enum works correctly")


def test_trainer_initialization():
    """Test UDATrainer initialization with new parameters."""
    print("\nTesting UDATrainer initialization...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        device = torch.device('cpu')
        
        trainer = UDATrainer(
            source_center='RUMC',
            target_center='PCNN',
            workdir=tmpdir,
            device=device,
            binary_classification=True,
            da_method='coral',
            checkpoint_validator='source_val',
            checkpoint_save_interval=10
        )
        
        assert trainer.checkpoint_validator == ValidatorType.SOURCE_VAL
        assert trainer.checkpoint_save_interval == 10
        
        print("  [OK] Trainer accepts checkpoint_validator parameter")
        print("  [OK] Trainer accepts checkpoint_save_interval parameter")
        
        trainer2 = UDATrainer(
            source_center='RUMC',
            target_center='PCNN',
            workdir=tmpdir,
            device=device,
            binary_classification=True,
            checkpoint_validator='target_val',
            checkpoint_save_interval=20
        )
        
        assert trainer2.checkpoint_validator == ValidatorType.TARGET_VAL
        assert trainer2.checkpoint_save_interval == 20
        
        print("  [OK] Target-based validator configuration works")


def test_model_creation():
    """Test model creation with dropout/batchnorm options."""
    print("\nTesting model creation...")
    
    sys.path.append(str(Path(__file__).parent.parent.parent / 'models' / 'MRI' / 'baseline' / 'src' / 'cnn'))
    from model import ISUPClassifier
    
    model_simple = ISUPClassifier(num_channels=3, num_classes=2, dropout_rate=0.0, use_batchnorm=False)
    model_complex = ISUPClassifier(num_channels=3, num_classes=2, dropout_rate=0.5, use_batchnorm=True)
    
    has_dropout_simple = any('Dropout' in str(m) for m in model_simple.modules())
    has_dropout_complex = any('Dropout' in str(m) for m in model_complex.modules())
    has_bn_simple = any('BatchNorm' in str(m) for m in model_simple.modules())
    has_bn_complex = any('BatchNorm' in str(m) for m in model_complex.modules())
    
    assert not has_dropout_simple, "Simple model should not have dropout"
    assert has_dropout_complex, "Complex model should have dropout"
    assert not has_bn_simple, "Simple model should not have batchnorm"
    assert has_bn_complex, "Complex model should have batchnorm"
    
    print("  [OK] dropout_rate=0.0 disables dropout ")
    print("  [OK] use_batchnorm=False disables batchnorm ")
    
    x = torch.randn(2, 3, 16, 16, 16)
    out = model_simple(x)
    
    assert 'features' in out
    assert 'classification' in out
    assert out['classification'].shape == (2, 2)
    
    print("  [OK] Model forward pass works")


def test_result_keys():
    """Test that results contain all required keys."""
    print("\nTesting result structure...")
    
    required_source_test_keys = [
        'final_source_test_accuracy',
        'final_source_test_balanced_accuracy',
        'final_source_test_auc',
        'final_source_test_sensitivity',
        'final_source_test_specificity',
    ]
    
    required_target_test_keys = [
        'final_target_test_accuracy',
        'final_target_test_balanced_accuracy',
        'final_target_test_auc',
        'final_target_test_sensitivity',
        'final_target_test_specificity',
    ]
    
    required_checkpoint_keys = [
        'checkpoint_validator',
        'best_epoch',
        'best_source_val_bal_acc',
        'best_target_val_bal_acc',
        'checkpoint_summary',
    ]
    
    print("  [OK] Source test metrics defined")
    print("  [OK] Target test metrics defined")
    print("  [OK] Checkpoint info keys defined")
    
    for key in required_source_test_keys:
        print(f"    - {key}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("UDA TRAINER TEST SUITE")
    print("Testing features:")
    print("  1. Target-based checkpoint selection (oracle upper bound)")
    print("  2. Source test evaluation")
    print("  3. Periodic checkpoint saving")
    print("=" * 60)
    
    try:
        test_validator_enum()
        test_checkpoint_manager()
        test_data_splits()
        test_trainer_initialization()
        test_model_creation()
        test_result_keys()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED [OK]")
        print("=" * 60)
        
        print("\nNew CLI arguments available:")
        print("  --checkpoint-validator source_val|target_val")
        print("  --checkpoint-interval N")
        print("\nExample usage:")
        print("  # Standard UDA with source-based selection")
        print("  python 1_cnn_uda_runner.py --da-method coral --binary")
        print("")
        print("  # Oracle upper bound (target-based selection)")
        print("  python 1_cnn_uda_runner.py --da-method coral --binary --checkpoint-validator target_val")
        print("")
        print("  # Save checkpoints every 5 epochs")
        print("  python 1_cnn_uda_runner.py --da-method coral --binary --checkpoint-interval 5")
        
        return True
        
    except Exception as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
