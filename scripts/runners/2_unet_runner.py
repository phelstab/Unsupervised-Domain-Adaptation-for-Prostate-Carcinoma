# UV Virtual Environment: .venv-unet
# Setup: uv venv .venv-unet
# Install PyTorch: uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# Install dependencies: uv pip install picai-baseline nibabel SimpleITK scikit-image pandas tqdm tensorboard
# Data location: input/mri_data/Task2203/ (imagesTr/, labelsTr/)
# Run: python -m uv run --python .venv-unet python scripts\runners\2_unet_runner.py --weights_dir workdir\results\UNet\weights --overviews_dir workdir\results\UNet\overviews\Task2203_picai_baseline --folds 0

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='U-Net Runner (Windows, No Docker)')
    
    parser.add_argument('--weights_dir', type=str, required=True,
                       help='Path to export model checkpoints')
    parser.add_argument('--overviews_dir', type=str, required=True,
                       help='Base path to training/validation data sheets')
    parser.add_argument('--folds', nargs='+', default=['0'],
                       help='Folds selected for training/validation run')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Mini-batch size (reduced for Windows)')
    parser.add_argument('--max_threads', type=int, default=2,
                       help='Max threads/workers for data loaders (Windows-safe default)')
    parser.add_argument('--enable_da', type=int, default=1,
                       help='Enable data augmentation')
    parser.add_argument('--base_lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--focal_loss_gamma', type=float, default=1.0,
                       help='Focal Loss gamma value')
    parser.add_argument('--validate_n_epochs', type=int, default=10,
                       help='Trigger validation every N epochs')
    parser.add_argument('--validate_min_epoch', type=int, default=50,
                       help='Trigger validation after minimum N epochs')
    
    args = parser.parse_args()
    
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    
    workspace_root = Path(__file__).parent.parent.parent
    train_script = workspace_root / 'models' / 'MRI' / 'baseline' / 'src' / 'unet' / 'train.py'
    
    if not train_script.exists():
        raise FileNotFoundError(f'U-Net train script not found at {train_script}')
    
    cmd = [
        sys.executable, str(train_script),
        '--weights_dir', os.path.abspath(args.weights_dir),
        '--overviews_dir', os.path.abspath(args.overviews_dir),
        '--folds', *args.folds,
        '--num_epochs', str(args.num_epochs),
        '--batch_size', str(args.batch_size),
        '--max_threads', str(args.max_threads),
        '--enable_da', str(args.enable_da),
        '--base_lr', str(args.base_lr),
        '--focal_loss_gamma', str(args.focal_loss_gamma),
        '--validate_n_epochs', str(args.validate_n_epochs),
        '--validate_min_epoch', str(args.validate_min_epoch),
    ]
    
    print(f'Running U-Net training with command:')
    print(' '.join(cmd))
    print()
    
    subprocess.check_call(cmd)

if __name__ == '__main__':
    main()
