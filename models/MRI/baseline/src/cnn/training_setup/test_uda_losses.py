#!/usr/bin/env python3
"""
Test script for UDA loss functions (MMD, DANN, CORAL, Entropy).
Run: python -m uv run --python .venv-cnn models/MRI/baseline/src/cnn/training_setup/test_uda_losses.py
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training_setup.loss_functions import (
    MMDLoss, DANNLoss, CORALLoss, EntropyLoss, MCCLoss,
    GradientReversalLayer, revgrad
)


def test_mmd_loss():
    """Test MMD loss computation."""
    print("=" * 50)
    print("Testing MMDLoss")
    print("=" * 50)
    
    mmd_loss = MMDLoss()  # Using default TLlib-style parameters
    
    # Test 1: Same distribution should have low MMD
    source = torch.randn(32, 512)
    target = source.clone()  # Identical
    loss_same = mmd_loss(source, target)
    print(f"[TEST 1] Same distribution MMD: {loss_same.item():.6f} (should be ~0)")
    assert loss_same.item() < 0.01, "Same distribution should have near-zero MMD"
    
    # Test 2: Different distributions should have higher MMD
    source = torch.randn(32, 512)
    target = torch.randn(32, 512) + 5.0  # Shifted distribution
    loss_diff = mmd_loss(source, target)
    print(f"[TEST 2] Different distribution MMD: {loss_diff.item():.6f} (should be > 0)")
    assert loss_diff.item() > 0.1, "Different distributions should have positive MMD"
    
    # Test 3: Gradient flows
    source = torch.randn(16, 512, requires_grad=True)
    target = torch.randn(16, 512, requires_grad=True)
    loss = mmd_loss(source, target)
    loss.backward()
    print(f"[TEST 3] Gradient flow check - source grad norm: {source.grad.norm().item():.6f}")
    assert source.grad is not None, "Gradient should flow to source"
    assert target.grad is not None, "Gradient should flow to target"
    
    # Test 4: Larger batch
    source = torch.randn(64, 512)
    target = torch.randn(64, 512)
    loss = mmd_loss(source, target)
    print(f"[TEST 4] Larger batch (64): {loss.item():.6f}")
    
    # Test 5: Different batch sizes (critical for last batch in DataLoader)
    source = torch.randn(16, 512)
    target = torch.randn(4, 512)  # Smaller target batch
    loss = mmd_loss(source, target)
    print(f"[TEST 5] Different batch sizes (16 vs 4): {loss.item():.6f}")
    
    # Test 6: Reverse different batch sizes
    source = torch.randn(8, 512)
    target = torch.randn(32, 512)
    loss = mmd_loss(source, target)
    print(f"[TEST 6] Different batch sizes (8 vs 32): {loss.item():.6f}")
    
    print("[PASS] MMDLoss tests passed\n")


def test_gradient_reversal():
    """Test gradient reversal layer."""
    print("=" * 50)
    print("Testing GradientReversalLayer")
    print("=" * 50)
    
    # Test 1: Forward pass should be identity
    grl = GradientReversalLayer(alpha=1.0)
    x = torch.tensor([1.0, 2.0, 3.0])
    y = grl(x)
    print(f"[TEST 1] Forward pass: input={x.tolist()}, output={y.tolist()}")
    assert torch.allclose(x, y), "Forward pass should be identity"
    
    # Test 2: Backward pass should reverse gradient
    x = torch.tensor([4.0], requires_grad=True)
    alpha = torch.tensor(1.0)
    
    # Normal gradient
    y_normal = x * 5
    y_normal.backward()
    grad_normal = x.grad.clone()
    
    # Reversed gradient
    x_rev = torch.tensor([4.0], requires_grad=True)
    y_rev = revgrad(x_rev * 5, alpha)
    y_rev.backward()
    grad_reversed = x_rev.grad.clone()
    
    print(f"[TEST 2] Normal gradient: {grad_normal.item()}, Reversed: {grad_reversed.item()}")
    assert grad_normal.item() == -grad_reversed.item(), "Gradients should be negated"
    
    # Test 3: Alpha scaling
    x = torch.tensor([4.0], requires_grad=True)
    alpha = torch.tensor(2.0)
    y = revgrad(x * 5, alpha)
    y.backward()
    print(f"[TEST 3] Alpha=2.0 gradient: {x.grad.item()} (should be -10)")
    assert x.grad.item() == -10.0, "Alpha should scale the reversed gradient"
    
    print("[PASS] GradientReversalLayer tests passed\n")


def test_dann_loss():
    """Test DANN loss computation."""
    print("=" * 50)
    print("Testing DANNLoss")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dann_loss = DANNLoss(feature_dim=512, hidden_dim=256).to(device)
    
    # Test 1: Basic forward pass
    source = torch.randn(16, 512, device=device)
    target = torch.randn(16, 512, device=device)
    loss = dann_loss(source, target)
    print(f"[TEST 1] DANN loss value: {loss.item():.6f}")
    assert loss.item() > 0, "Loss should be positive"
    
    # Test 2: Alpha scheduling
    dann_loss.set_alpha(0.5, device=device)
    loss_half = dann_loss(source, target)
    print(f"[TEST 2] DANN loss with alpha=0.5: {loss_half.item():.6f}")
    
    # Test 3: Gradient flow with feature extractor simulation
    feature_extractor = nn.Linear(256, 512).to(device)
    classifier = nn.Linear(512, 2).to(device)
    
    source_input = torch.randn(8, 256, device=device)
    target_input = torch.randn(8, 256, device=device)
    
    source_features = feature_extractor(source_input)
    target_features = feature_extractor(target_input)
    
    dann_loss.set_alpha(1.0, device=device)
    loss = dann_loss(source_features, target_features)
    loss.backward()
    
    print(f"[TEST 3] Feature extractor grad norm: {feature_extractor.weight.grad.norm().item():.6f}")
    assert feature_extractor.weight.grad is not None, "Gradient should flow to feature extractor"
    
    # Test 4: Alpha schedule simulation (like in training)
    print("[TEST 4] Alpha schedule over epochs:")
    import numpy as np
    for epoch in [0, 25, 50, 75, 100]:
        p = epoch / 100
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        print(f"  Epoch {epoch:3d}: alpha = {alpha:.4f}")
    
    print("[PASS] DANNLoss tests passed\n")


def test_coral_loss():
    """Test CORAL loss computation."""
    print("=" * 50)
    print("Testing CORALLoss")
    print("=" * 50)
    
    coral_loss = CORALLoss()
    
    # Test 1: Same covariance should have zero loss
    source = torch.randn(32, 512)
    target = source.clone()
    loss = coral_loss(source, target)
    print(f"[TEST 1] Same covariance CORAL: {loss.item():.6f} (should be ~0)")
    assert loss.item() < 0.01, "Same covariance should have near-zero CORAL"
    
    # Test 2: Different covariance
    source = torch.randn(32, 512)
    target = torch.randn(32, 512) * 2  # Different scale
    loss = coral_loss(source, target)
    print(f"[TEST 2] Different covariance CORAL: {loss.item():.6f}")
    
    print("[PASS] CORALLoss tests passed\n")


def test_entropy_loss():
    """Test Entropy loss computation."""
    print("=" * 50)
    print("Testing EntropyLoss")
    print("=" * 50)
    
    entropy_loss = EntropyLoss()
    
    # Test 1: Confident predictions (low entropy)
    logits_confident = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    loss_confident = entropy_loss(logits_confident)
    print(f"[TEST 1] Confident predictions entropy: {loss_confident.item():.6f} (should be ~0)")
    
    # Test 2: Uncertain predictions (high entropy)
    logits_uncertain = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    loss_uncertain = entropy_loss(logits_uncertain)
    print(f"[TEST 2] Uncertain predictions entropy: {loss_uncertain.item():.6f} (should be ~0.69)")
    
    assert loss_confident < loss_uncertain, "Confident should have lower entropy"
    
    print("[PASS] EntropyLoss tests passed\n")


def test_integration():
    """Test integration of all losses in a training-like scenario."""
    print("=" * 50)
    print("Testing Integration (Training Simulation)")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(100, 512),
        nn.ReLU(),
        nn.Linear(512, 2)
    ).to(device)
    
    # Losses
    ce_loss = nn.CrossEntropyLoss()
    coral_loss = CORALLoss()
    entropy_loss = EntropyLoss()
    mmd_loss = MMDLoss()
    dann_loss = DANNLoss(feature_dim=512).to(device)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(dann_loss.parameters()),
        lr=0.001
    )
    
    # Simulate one training step
    source_data = torch.randn(16, 100, device=device)
    source_labels = torch.randint(0, 2, (16,), device=device)
    target_data = torch.randn(16, 100, device=device)
    
    optimizer.zero_grad()
    
    # Forward
    source_features = model[0](source_data)
    source_features = model[1](source_features)
    source_logits = model[2](source_features)
    
    target_features = model[0](target_data)
    target_features = model[1](target_features)
    target_logits = model[2](target_features)
    
    # Compute losses
    cls_loss = ce_loss(source_logits, source_labels)
    coral = coral_loss(source_features, target_features)
    entropy = entropy_loss(target_logits)
    mmd = mmd_loss(source_features, target_features)
    dann = dann_loss(source_features, target_features)
    
    total_loss = cls_loss + 0.5 * (coral + entropy + mmd + dann)
    total_loss.backward()
    optimizer.step()
    
    print(f"Classification loss: {cls_loss.item():.4f}")
    print(f"CORAL loss: {coral.item():.4f}")
    print(f"Entropy loss: {entropy.item():.4f}")
    print(f"MMD loss: {mmd.item():.4f}")
    print(f"DANN loss: {dann.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")
    
    print("[PASS] Integration test passed\n")


def test_mcc_loss():
    """Test MCC (Minimum Class Confusion) loss computation."""
    print("=" * 50)
    print("Testing MCCLoss (Minimum Class Confusion)")
    print("=" * 50)
    
    mcc_loss = MCCLoss(temperature=2.5)
    
    # Test 1: Confident predictions should have low MCC (low confusion)
    # Create one-hot like predictions (very confident)
    batch_size, num_classes = 32, 5
    logits_confident = torch.zeros(batch_size, num_classes)
    for i in range(batch_size):
        logits_confident[i, i % num_classes] = 10.0  # High logit for one class
    loss_confident = mcc_loss(logits_confident)
    print(f"[TEST 1] Confident predictions MCC: {loss_confident.item():.6f} (should be low)")
    
    # Test 2: Uniform predictions should have high MCC (high confusion)
    logits_uniform = torch.zeros(batch_size, num_classes)  # All equal = max confusion
    loss_uniform = mcc_loss(logits_uniform)
    print(f"[TEST 2] Uniform predictions MCC: {loss_uniform.item():.6f} (should be higher)")
    assert loss_uniform.item() > loss_confident.item(), \
        "Uniform predictions should have higher MCC than confident predictions"
    
    # Test 3: MCC is non-negative
    logits_random = torch.randn(batch_size, num_classes)
    loss_random = mcc_loss(logits_random)
    print(f"[TEST 3] Random predictions MCC: {loss_random.item():.6f} (should be >= 0)")
    assert loss_random.item() >= 0, "MCC should be non-negative"
    
    # Test 4: Gradient flows
    logits = torch.randn(16, num_classes, requires_grad=True)
    loss = mcc_loss(logits)
    loss.backward()
    print(f"[TEST 4] Gradient flow check - logits grad norm: {logits.grad.norm().item():.6f}")
    assert logits.grad is not None, "Gradient should flow to logits"
    
    # Test 5: Temperature effect
    mcc_high_temp = MCCLoss(temperature=5.0)
    mcc_low_temp = MCCLoss(temperature=1.0)
    logits = torch.randn(batch_size, num_classes)
    loss_high_temp = mcc_high_temp(logits)
    loss_low_temp = mcc_low_temp(logits)
    print(f"[TEST 5] High temp MCC: {loss_high_temp.item():.6f}, Low temp MCC: {loss_low_temp.item():.6f}")
    
    print("[PASS] MCCLoss tests passed\n")


def main():
    print("\n" + "=" * 50)
    print("UDA Loss Functions Test Suite")
    print("=" * 50 + "\n")
    
    try:
        test_mmd_loss()
        test_gradient_reversal()
        test_dann_loss()
        test_coral_loss()
        test_entropy_loss()
        test_mcc_loss()
        test_integration()
        
        print("=" * 50)
        print("ALL TESTS PASSED")
        print("=" * 50)
        return 0
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
