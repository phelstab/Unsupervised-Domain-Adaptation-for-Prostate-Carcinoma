# Model Runners - Windows Setup (No Docker)

This directory contains runner scripts for each baseline model. Each model uses a separate `uv` virtual environment to avoid dependency conflicts.

## Prerequisites

1. Install [uv](https://github.com/astral-sh/uv): `pip install uv`
2. Enable Windows long paths (optional but recommended):
   - Run `gpedit.msc`
   - Navigate to: Computer Configuration > Administrative Templates > System > Filesystem
   - Enable "Enable Win32 long paths"

## Virtual Environment Setup

### 1. U-Net Segmentation Model

**Environment**: `.venv-unet`

```powershell
# Create virtual environment
uv venv .venv-unet

# Install PyTorch (adjust CUDA version as needed)
# For CUDA 12.1:
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# For CPU only:
# uv pip install torch torchvision torchaudio

# Install dependencies
uv pip install picai-baseline nibabel SimpleITK scikit-image pandas tqdm tensorboard
```

**Run training**:
```powershell
uv run -p .venv-unet python scripts\runners\unet_runner.py --weights_dir workdir\results\UNet\weights --overviews_dir workdir\results\UNet\overviews --folds 0
```

### 2. nnU-Net Baseline Model

**Environment**: `.venv-nnunet`

```powershell
# Create virtual environment
uv venv .venv-nnunet

# Install PyTorch (adjust CUDA version as needed)
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Install nnU-Net and dependencies
uv pip install nnunet==1.7.0 batchgenerators scikit-image scipy nibabel SimpleITK scikit-learn tqdm

# Install PI-CAI dependencies
uv pip install picai-baseline picai-eval report-guided-annotation
```

**Run training**:
```powershell
uv run -p .venv-nnunet python scripts\runners\nnunet_runner.py --inputdir input --workdir workdir --folds 0 1 2 3 4
```

**Skip training/inference**:
```powershell
# Skip training (inference only)
uv run -p .venv-nnunet python scripts\runners\nnunet_runner.py --inputdir input --workdir workdir --skip_training

# Skip inference (training only)
uv run -p .venv-nnunet python scripts\runners\nnunet_runner.py --inputdir input --workdir workdir --skip_inference
```

### 3. nnDetection Model

**Environment**: `.venv-nndet`

**⚠️ CRITICAL: nnDetection does NOT support Windows!**

nnDetection officially only supports Linux. You have three options:

#### Option 1: WSL2 (Recommended)

Install and run nnDetection inside Windows Subsystem for Linux:

```powershell
# Install WSL2 (in Windows PowerShell)
wsl --install -d Ubuntu-22.04

# Start WSL2
wsl

# Inside WSL2, navigate to project
cd /mnt/c/workspace/uulm/PCa-classification

# Follow installation steps from install_dependencies.md
```

See [install_dependencies.md](file:///c:/workspace/uulm/PCa-classification/scripts/install_dependencies.md) for complete WSL2 setup instructions.

#### Option 2: Docker

```powershell
# Pull pre-built container
docker pull joeranbosma/picai_nndetection:latest

# Or build from source
cd models\MRI\baseline\src\nndetection\training_docker
docker build --tag picai_nndetection:latest .
```

#### Option 3: Skip nnDetection

Use only the three Windows-compatible baselines (U-Net, nnU-Net, CNN).

**Run training**:
```powershell
uv run -p .venv-nndet python scripts\runners\nndetection_runner.py --workdir workdir --fold 0
```

**Run prediction**:
```powershell
uv run -p .venv-nndet python scripts\runners\nndetection_runner.py --workdir workdir --fold 0 --predict_input workdir\nnDet_data\Task2201_picai_baseline\raw_splitted\imagesTs --predict_output workdir\nnDet_preds\fold0
```

### 4. CNN/ISUP Classifier

**Environment**: `.venv-cnn`

```powershell
# Create virtual environment
uv venv .venv-cnn

# Install PyTorch (adjust CUDA version as needed)
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Install dependencies
uv pip install numpy matplotlib pandas scikit-image nibabel SimpleITK tqdm tensorboard

# Install picai-baseline if needed
uv pip install picai-baseline
```

**Run training**:
```powershell
uv run -p .venv-cnn python scripts\runners\cnn_runner.py --weights_dir workdir\results\CNN\weights --folds 0
```

## Common Configuration Options

### U-Net Runner
- `--weights_dir`: Directory for model checkpoints
- `--overviews_dir`: Directory for training/validation data sheets
- `--folds`: Space-separated fold numbers (e.g., `0 1 2`)
- `--num_epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 4 for Windows)
- `--max_threads`: Data loader threads (default: 2 for Windows)

### nnU-Net Runner
- `--inputdir`: Input data directory (e.g., `input`)
- `--workdir`: Working directory (e.g., `workdir`)
- `--task`: Task name (default: `Task2201_picai_baseline`)
- `--trainer`: Trainer class (default: `nnUNetTrainerV2_Loss_FL_and_CE_checkpoints`)
- `--folds`: Space-separated fold numbers
- `--run_eval`: Enable evaluation (requires picai-eval)

### nnDetection Runner
- `--workdir`: Working directory
- `--task`: Task name
- `--model`: Model name (default: `RetinaUNetV001_D3V001_3d`)
- `--fold`: Fold number
- `--predict_input`: Input directory for prediction
- `--predict_output`: Output directory for prediction

### CNN Runner
- `--weights_dir`: Directory for model checkpoints
- `--overviews_dir`: Path to U-Net overviews
- `--marksheet_path`: Path to marksheet with ISUP grades
- `--folds`: Comma-separated fold numbers (e.g., `0,1,2,3,4`)
- `--num_epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 4 for Windows)

## Windows-Specific Optimizations

All runners include Windows-safe defaults:
- Reduced number of workers (`max_threads=2`, `num_workers=0`)
- Thread limiting (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`)
- Smaller batch sizes to avoid memory issues
- Absolute path conversions for Windows compatibility

## Environment Variables

### nnU-Net
The runner automatically sets these if using the v1 CLI fallback:
- `NNUNET_RAW_DATA_BASE`: `{workdir}\nnUNet_raw_data`
- `NNUNET_PREPROCESSED`: `{workdir}\nnUNet_preprocessed`
- `NNUNET_RESULTS`: `{workdir}\results\nnUNet`

### nnDetection
The runner automatically sets:
- `det_data`: `{workdir}\nnDet_data` (local staging)
- `det_models`: `{workdir}\results\nnDet` (model outputs)

## Troubleshooting

### GPU/CUDA Issues
- Verify PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Install correct CUDA version from [PyTorch website](https://pytorch.org/get-started/locally/)
- For CPU training, use CPU-only PyTorch (slower but more compatible)

### Path Length Issues
- Keep paths short (e.g., `D:\picai\work` instead of deeply nested directories)
- Enable Windows long paths as described above

### Multiprocessing Errors
- Reduce `--max_threads` or `--num_workers` to 0 or 1
- Use `--batch_size 1` or `2` if memory errors occur

### nnDetection on Windows
- If installation fails, nnDetection may not have Windows wheels
- Consider using WSL2 for a Linux environment
- Alternatively, use CPU PyTorch for development/testing

## Directory Structure

Expected project structure:
```
PCa-classification/
├── input/                          # Input data
│   ├── images/
│   └── picai_labels/
├── workdir/                        # Working directory
│   ├── results/
│   │   ├── UNet/
│   │   ├── nnUNet/
│   │   ├── CNN/
│   │   └── nnDet/
│   ├── nnUNet_raw_data/
│   └── nnDet_data/
├── models/MRI/baseline/            # Baseline models source
└── scripts/runners/                # This directory
    ├── unet_runner.py
    ├── nnunet_runner.py
    ├── nndetection_runner.py
    ├── cnn_runner.py
    └── README.md
```

## Quick Start Example

```powershell
# 1. Train U-Net for fold 0
uv run -p .venv-unet python scripts\runners\unet_runner.py `
  --weights_dir workdir\results\UNet\weights `
  --overviews_dir workdir\results\UNet\overviews `
  --folds 0 `
  --num_epochs 50 `
  --batch_size 2

# 2. Train CNN classifier using U-Net predictions
uv run -p .venv-cnn python scripts\runners\cnn_runner.py `
  --weights_dir workdir\results\CNN\weights `
  --overviews_dir workdir\results\UNet\overviews\Task2203_picai_baseline `
  --folds 0 `
  --num_epochs 50

# 3. Train nnU-Net (all folds)
uv run -p .venv-nnunet python scripts\runners\nnunet_runner.py `
  --inputdir input `
  --workdir workdir `
  --folds 0 1 2 3 4
```
