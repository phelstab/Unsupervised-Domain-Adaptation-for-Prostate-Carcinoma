"""
Neural network selector for ISUP Classification CNN
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from model import create_isup_classifier


def isup_network_for_run(args, device):
    """
    Create ISUP classification network based on training arguments
    
    Args:
        args: Training arguments
        device: Device to run on (cuda/cpu)
    
    Returns:
        ISUP classification model
    """
    
    model = create_isup_classifier(
        num_channels=args.num_channels,
        num_classes=args.num_classes,
        features=args.model_features,
        strides=args.model_strides,
        dropout_rate=args.dropout_rate
    )
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"ISUP Classifier created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")
    
    return model
