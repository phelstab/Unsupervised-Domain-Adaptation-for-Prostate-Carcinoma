#!/usr/bin/env python3
"""
Test script for BNM (Batch Nuclear-norm Maximization) loss function.
Run: python -m uv run --python .venv-cnn models/MRI/baseline/src/cnn/training_setup/test_bnm_loss.py
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training_setup.loss_functions import BNMLoss


def test_bnm_loss():
    """Test BNM loss computation."""
    print("=" * 50)
    print("Testing BNMLoss (Batch Nuclear-norm Maximization)")
    print("=" * 50)
    
    bnm_loss = BNMLoss()
    
    # Test 1: One-hot predictions should have maximum nuclear norm (low loss)
    # When each sample predicts a different class with confidence, nuclear norm is high
    print("\n[TEST 1] One-hot diverse predictions (ideal case)")
    logits_onehot = torch.zeros(4, 4)
    for i in range(4):
        logits_onehot[i, i] = 10.0  # Each sample predicts different class
    loss_onehot = bnm_loss(logits_onehot)
    print(f"  One-hot diverse logits loss: {loss_onehot.item():.6f}")
    print(f"  (Negative value expected - we maximize nuclear norm)")
    
    # Test 2: Uniform predictions should have lower nuclear norm (higher loss)
    print("\n[TEST 2] Uniform predictions (worst case)")
    logits_uniform = torch.zeros(4, 4)  # All zeros -> uniform after softmax
    loss_uniform = bnm_loss(logits_uniform)
    print(f"  Uniform logits loss: {loss_uniform.item():.6f}")
    
    # Test 3: All same prediction (low diversity)
    print("\n[TEST 3] All same prediction (no diversity)")
    logits_same = torch.zeros(4, 4)
    logits_same[:, 0] = 10.0  # All samples predict class 0
    loss_same = bnm_loss(logits_same)
    print(f"  All same class logits loss: {loss_same.item():.6f}")
    
    # Verify ranking: one-hot < uniform (one-hot is better = lower loss)
    # and one-hot < same
    print("\n[TEST 4] Verify loss ranking")
    print(f"  One-hot loss: {loss_onehot.item():.6f}")
    print(f"  Uniform loss: {loss_uniform.item():.6f}")
    print(f"  All-same loss: {loss_same.item():.6f}")
    
    # BNM loss is negative, so lower = better (higher nuclear norm)
    assert loss_onehot.item() < loss_uniform.item(), \
        f"One-hot should have lower loss than uniform: {loss_onehot.item():.6f} vs {loss_uniform.item():.6f}"
    print("  [PASS] One-hot < Uniform (diverse confident predictions are better)")
    
    # Test 5: Gradient flows
    print("\n[TEST 5] Gradient flow test")
    logits = torch.randn(16, 6, requires_grad=True)
    loss = bnm_loss(logits)
    loss.backward()
    assert logits.grad is not None, "Gradient should flow to input"
    assert not torch.isnan(logits.grad).any(), "Gradient should not contain NaN"
    print(f"  Gradient shape: {logits.grad.shape}")
    print(f"  Gradient norm: {logits.grad.norm().item():.6f}")
    print("  [PASS] Gradients flow correctly")
    
    # Test 6: Different batch sizes
    print("\n[TEST 6] Different batch sizes")
    for batch_size in [1, 8, 32, 64]:
        logits = torch.randn(batch_size, 6)
        loss = bnm_loss(logits)
        print(f"  Batch size {batch_size:3d}: loss = {loss.item():.6f}")
    print("  [PASS] Works with different batch sizes")
    
    # Test 7: Different number of classes
    print("\n[TEST 7] Different number of classes")
    for num_classes in [2, 6, 10]:
        logits = torch.randn(16, num_classes)
        loss = bnm_loss(logits)
        print(f"  {num_classes} classes: loss = {loss.item():.6f}")
    print("  [PASS] Works with different number of classes")
    
    print("\n" + "=" * 50)
    print("[PASS] All BNMLoss tests passed!")
    print("=" * 50)


def test_bnm_integration():
    """Test BNM loss in a training-like scenario."""
    print("\n" + "=" * 50)
    print("Testing BNM Integration (Training Simulation)")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(100, 512),
        nn.ReLU(),
        nn.Linear(512, 2)  # Binary classification
    ).to(device)
    
    # Losses
    ce_loss = nn.CrossEntropyLoss()
    bnm_loss = BNMLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Simulate one training step
    source_data = torch.randn(16, 100, device=device)
    source_labels = torch.randint(0, 2, (16,), device=device)
    target_data = torch.randn(16, 100, device=device)
    
    optimizer.zero_grad()
    
    # Forward
    source_logits = model(source_data)
    target_logits = model(target_data)
    
    # Compute losses
    cls_loss = ce_loss(source_logits, source_labels)
    bnm = bnm_loss(target_logits)
    
    # Combined loss (cls_loss + da_weight * bnm_loss)
    da_weight = 1.0
    total_loss = cls_loss + da_weight * bnm
    
    # Backward
    total_loss.backward()
    optimizer.step()
    
    print(f"Classification loss: {cls_loss.item():.4f}")
    print(f"BNM loss: {bnm.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")
    
    print("\n[PASS] Integration test passed!")


if __name__ == "__main__":
    test_bnm_loss()
    test_bnm_integration()
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
