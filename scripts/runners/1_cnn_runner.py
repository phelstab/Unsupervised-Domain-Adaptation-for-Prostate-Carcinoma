# UV Virtual Environment: .venv-cnn
# Setup: uv venv .venv-cnn
# Install PyTorch: uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# Install dependencies: uv pip install numpy matplotlib pandas scikit-image nibabel SimpleITK tqdm tensorboard
# Install picai-baseline if needed: uv pip install picai-baseline
# Run: python -m uv run --python .venv-cnn python scripts\runners\1_cnn_runner.py --weights_dir workdir\results\CNN\weights --folds 0 --num_epochs 5

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='CNN/ISUP Classifier Runner (Windows, No Docker)')
    
    parser.add_argument('--weights_dir', type=str, required=True,
                       help='Path to export model checkpoints')
    parser.add_argument('--overviews_dir', type=str, 
                       default='workdir/results/UNet/overviews/Task2203_picai_baseline',
                       help='Path to U-Net overviews (patient-level predictions)')
    parser.add_argument('--marksheet_path', type=str,
                       default='input/picai_labels/clinical_information/marksheet.csv',
                       help='Path to marksheet with ISUP grades')
    parser.add_argument('--folds', type=str, default='0',
                       help='Comma-separated folds (e.g., 0,1,2,3,4)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Mini-batch size (reduced for Windows)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    
    workspace_root = Path(__file__).parent.parent.parent
    train_script = workspace_root / 'models' / 'MRI' / 'baseline' / 'src' / 'cnn' / 'train.py'
    
    if not train_script.exists():
        raise FileNotFoundError(f'CNN train script not found at {train_script}')
    
    cmd = [
        sys.executable, str(train_script),
        '--weights_dir', os.path.abspath(args.weights_dir),
        '--overviews_dir', os.path.abspath(args.overviews_dir),
        '--marksheet_path', os.path.abspath(args.marksheet_path),
        '--folds', args.folds,
        '--num_epochs', str(args.num_epochs),
        '--batch_size', str(args.batch_size),
        '--learning_rate', str(args.learning_rate),
        '--patience', str(args.patience),
    ]
    
    print(f'Running CNN/ISUP training with command:')
    print(' '.join(cmd))
    print()
    
    subprocess.check_call(cmd)

if __name__ == '__main__':
    main()
