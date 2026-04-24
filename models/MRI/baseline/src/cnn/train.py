#!/usr/bin/env python3

"""
Training script for ISUP Classification CNN
Usage: python train.py --weights_dir results/CNN/weights --folds 0 --num_epochs 50
"""

import argparse
import os
import sys
import time
from pathlib import Path
import json
import torch
import torch.optim as optimizer
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt

# Add the parent directories to the path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from training_setup.hyperparams import get_isup_hyperparams
from training_setup.neural_network_selector import isup_network_for_run
from training_setup.data_generator import prepare_isup_datagens_direct, create_isup_args_from_unet_args
from training_setup.loss_functions import create_isup_loss, calculate_class_weights


class ISUPTrainer:
    """ISUP Classification trainer"""
    
    def __init__(self, args):
        self.args = args
        
        # Try CUDA with actual operation test for RTX 5090
        if torch.cuda.is_available():
            try:
                # Test actual CUDA operations including batchnorm
                test_tensor = torch.randn(1, 3, 4, 4, 4).cuda()
                bn = torch.nn.BatchNorm3d(3).cuda()
                _ = bn(test_tensor)
                del test_tensor, bn
                torch.cuda.empty_cache()
                self.device = torch.device('cuda')
                print("CUDA is working")
            except RuntimeError as e:
                print(f"WARNING: CUDA test failed: {str(e)[:80]}")
                print("Using CPU instead (will be slower)")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        
        print(f"Using device: {self.device}")
        
        # Create directories
        self.weights_dir = Path(args.weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup model
        self.model = isup_network_for_run(args, self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=args.patience//2
        )
        
        # Training state
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, loss_fn):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_seg_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            data = batch['data'].to(self.device)
            isup_targets = batch['isup_grade'].to(self.device)
            
            targets = {'isup_grade': isup_targets}
            if 'seg' in batch:
                targets['seg'] = batch['seg'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            losses = loss_fn(outputs, targets)
            
            # Backward pass
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += losses['total_loss'].item()
            total_cls_loss += losses['classification_loss'].item()
            if 'segmentation_loss' in losses:
                total_seg_loss += losses['segmentation_loss'].item()
            
            # Accuracy
            predicted = torch.argmax(outputs['classification'], dim=1)
            correct += (predicted == isup_targets).sum().item()
            total += isup_targets.size(0)
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {losses["total_loss"].item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        avg_cls_loss = total_cls_loss / len(train_loader)
        avg_seg_loss = total_seg_loss / len(train_loader) if total_seg_loss > 0 else 0
        accuracy = 100.0 * correct / total
        
        return {
            'total_loss': avg_loss,
            'classification_loss': avg_cls_loss,
            'segmentation_loss': avg_seg_loss,
            'accuracy': accuracy
        }
    
    def validate(self, val_loader, loss_fn):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_seg_loss = 0.0
        correct = 0
        total = 0
        
        # For confusion matrix
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                data = batch['data'].to(self.device)
                isup_targets = batch['isup_grade'].to(self.device)
                
                targets = {'isup_grade': isup_targets}
                if 'seg' in batch:
                    targets['seg'] = batch['seg'].to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                losses = loss_fn(outputs, targets)
                
                # Statistics
                total_loss += losses['total_loss'].item()
                total_cls_loss += losses['classification_loss'].item()
                if 'segmentation_loss' in losses:
                    total_seg_loss += losses['segmentation_loss'].item()
                
                # Accuracy
                predicted = torch.argmax(outputs['classification'], dim=1)
                correct += (predicted == isup_targets).sum().item()
                total += isup_targets.size(0)
                
                # Store for confusion matrix
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(isup_targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        avg_cls_loss = total_cls_loss / len(val_loader)
        avg_seg_loss = total_seg_loss / len(val_loader) if total_seg_loss > 0 else 0
        accuracy = 100.0 * correct / total
        
        return {
            'total_loss': avg_loss,
            'classification_loss': avg_cls_loss,
            'segmentation_loss': avg_seg_loss,
            'accuracy': accuracy,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def save_checkpoint(self, epoch, fold_id, val_results):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'args': vars(self.args),
            'val_results': val_results
        }
        
        # Save best model
        if val_results['total_loss'] < self.best_loss:
            self.best_loss = val_results['total_loss']
            torch.save(checkpoint, self.weights_dir / f'isup_cnn_F{fold_id}_best.pt')
            print(f"  ✓ New best model saved (loss: {self.best_loss:.4f})")
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Save latest model
        torch.save(checkpoint, self.weights_dir / f'isup_cnn_F{fold_id}_latest.pt')
    
    def plot_training_curves(self, fold_id):
        """Plot training curves"""
        if len(self.train_losses) == 0:
            return
        
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        train_epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(train_epochs, self.train_losses, 'b-', label='Training Loss')
        
        if len(self.val_losses) > 0:
            val_epochs = range(1, len(self.val_losses) + 1)
            ax1.plot(val_epochs, self.val_losses, 'r-', label='Validation Loss')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        if len(self.val_accuracies) > 0:
            val_epochs = range(1, len(self.val_accuracies) + 1)
            ax2.plot(val_epochs, self.val_accuracies, 'g-', label='Validation Accuracy')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.weights_dir / f'training_curves_F{fold_id}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self, fold_id=0):
        """Main training loop"""
        print(f"\n=== Training ISUP CNN - Fold {fold_id} ===")
        
        # Prepare data
        print("Loading data...")
        try:
            train_loader, val_loader, _ = prepare_isup_datagens_direct(
                self.args, 
                fold_id=fold_id,
                marksheet_path=getattr(self.args, 'marksheet_path', 'input/picai_labels/clinical_information/marksheet.csv')
            )
        except Exception as e:
            import traceback
            print(f"Error loading data: {e}")
            traceback.print_exc()
            return
        
        # Calculate class weights
        print("Calculating class weights...")
        train_targets = []
        for batch in train_loader:
            train_targets.extend(batch['isup_grade'].numpy())
        
        train_targets = torch.tensor(train_targets)
        class_weights = calculate_class_weights(train_targets, self.args.num_classes)
        class_weights = class_weights.to(self.device)
        
        print(f"Class weights: {class_weights.cpu()}")
        
        # Create loss function
        loss_fn = create_isup_loss(
            num_classes=self.args.num_classes,
            class_weights=class_weights,
            device=self.device
        )
        
        print(f"Starting training for {self.args.num_epochs} epochs...")
        
        for epoch in range(1, self.args.num_epochs + 1):
            start_time = time.time()
            
            print(f"\nEpoch {epoch}/{self.args.num_epochs}")
            print("-" * 50)
            
            # Training
            train_results = self.train_epoch(train_loader, loss_fn)
            self.train_losses.append(train_results['total_loss'])
            
            # Validation
            if epoch >= self.args.validate_min_epoch and epoch % self.args.validate_n_epochs == 0:
                print("  Validating...")
                val_results = self.validate(val_loader, loss_fn)
                self.val_losses.append(val_results['total_loss'])
                self.val_accuracies.append(val_results['accuracy'])
                
                # Learning rate scheduling
                self.scheduler.step(val_results['total_loss'])
                
                # Save checkpoint
                self.save_checkpoint(epoch, fold_id, val_results)
                
                # Print results
                epoch_time = time.time() - start_time
                print(f"  Train Loss: {train_results['total_loss']:.4f}, "
                      f"Train Acc: {train_results['accuracy']:.2f}%")
                print(f"  Val Loss: {val_results['total_loss']:.4f}, "
                      f"Val Acc: {val_results['accuracy']:.2f}%")
                print(f"  Time: {epoch_time:.1f}s, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Early stopping
                if self.patience_counter >= self.args.patience:
                    print(f"  Early stopping triggered (patience: {self.args.patience})")
                    break
            
            else:
                # Just print training results
                epoch_time = time.time() - start_time
                print(f"  Train Loss: {train_results['total_loss']:.4f}, "
                      f"Train Acc: {train_results['accuracy']:.2f}%, "
                      f"Time: {epoch_time:.1f}s")
        
        # Plot training curves
        self.plot_training_curves(fold_id)
        
        print(f"\n=== Training completed for fold {fold_id} ===")
        print(f"Best validation loss: {self.best_loss:.4f}")
        print(f"Model saved to: {self.weights_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train ISUP Classification CNN')
    
    # Required arguments
    parser.add_argument('--weights_dir', type=str, required=True,
                       help='Directory to save model weights')
    
    # Optional arguments
    parser.add_argument('--overviews_dir', type=str, 
                       default='workdir/results/UNet/overviews/Task2203_picai_baseline',
                       help='Directory containing data overviews')
    parser.add_argument('--marksheet_path', type=str,
                       default='input/picai_labels/clinical_information/marksheet.csv',
                       help='Path to marksheet with ISUP labels')
    parser.add_argument('--folds', type=str, default='0',
                       help='Comma-separated fold IDs to train (e.g., "0,1,2")')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    args.weights_dir = os.path.abspath(args.weights_dir)
    args.overviews_dir = os.path.abspath(args.overviews_dir)
    args.marksheet_path = os.path.abspath(args.marksheet_path)
    
    # Apply hyperparameters
    args = get_isup_hyperparams(args)
    
    print("ISUP CNN Training Configuration:")
    print(f"  Weights directory: {args.weights_dir}")
    print(f"  Marksheet path: {args.marksheet_path}")
    print(f"  Folds: {args.folds}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    
    # Parse folds
    fold_ids = [int(f.strip()) for f in args.folds.split(',')]
    
    # Train each fold
    for fold_id in fold_ids:
        try:
            trainer = ISUPTrainer(args)
            trainer.train(fold_id=fold_id)
        except Exception as e:
            print(f"Error training fold {fold_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== All folds completed ===")


if __name__ == '__main__':
    main()
