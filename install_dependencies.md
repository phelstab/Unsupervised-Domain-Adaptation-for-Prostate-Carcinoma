# Installation Commands for Virtual Environments

## Get Data 

1	scripts\setup\setup_directories.bat	Creates folder structure
2	scripts\setup\download.py	Downloads PI-CAI zips from Zenodo + extracts to input\images\
3	cd input && git clone https://github.com/DIAGNijmegen/picai_labels	Gets labels/marksheet
4	scripts\setup\flatten_images.py	Flattens input\images\ → input\images_flat\ // not needed anymore
5	(automatic)	Preprocessing (resample to 20×256×256, z-normalize → .npy) runs auto on first training launch
6	scripts\runners\uda_bats\main.bat	Run experiments


**System Info:**
- GPU: NVIDIA GeForce RTX 5090
- CUDA Compute Capability: **sm_120** (Blackwell architecture)
- System CUDA Version: 13.0
- PyTorch CUDA Version: **cu128** (CUDA 12.8 nightly)

## RTX 5090 (sm_120) Compatibility Notice

**IMPORTANT**: The RTX 5090 uses CUDA compute capability **sm_120** (Blackwell architecture), which is NOT supported by stable PyTorch releases as of November 2024.

**Solution**: Use PyTorch **nightly builds** with CUDA 12.8 support:

```bash
# Works with RTX 5090 (sm_120)
--index-url https://download.pytorch.org/whl/nightly/cu128

# Does NOT work with RTX 5090
--index-url https://download.pytorch.org/whl/cu124  # Stable release
```

**Symptoms of incompatibility**:
- Warning: `CUDA capability sm_120 is not compatible with current PyTorch`
- Runtime error: `CUDA error: no kernel image is available for execution on the device`
- Training crashes during forward pass or BatchNorm operations

All virtual environments below are configured with PyTorch nightly (cu128) for RTX 5090 support.

---

All virtual environments have been created. Run these commands to install dependencies:

## 1. U-Net Environment (.venv-unet) - DONE

```powershell
# Install PyTorch (nightly for RTX 5090 sm_120 support)
python -m uv pip install --python .venv-unet --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install dependencies
python -m uv pip install --python .venv-unet picai-baseline nibabel SimpleITK scikit-image pandas tqdm tensorboard batchgenerators monai openpyxl

# Verify RTX 5090 compatibility
python -m uv run --python .venv-unet python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); test = torch.randn(1,3,4,4,4).cuda(); bn = torch.nn.BatchNorm3d(3).cuda(); out = bn(test); print('RTX 5090 WORKS!')"
```

## 2. nnU-Net Environment (.venv-nnunet) - DONE

```powershell
# Install PyTorch (nightly for RTX 5090 sm_120 support)
python -m uv pip install --python .venv-nnunet --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install dependencies
python -m uv pip install --python .venv-nnunet nnunet==1.7.1 batchgenerators scikit-image scipy nibabel SimpleITK scikit-learn tqdm picai-baseline picai-eval report-guided-annotation monai

# Copy custom trainers to nnUNet installation (REQUIRED for PI-CAI baseline)
# These custom trainers implement Focal Loss + Cross Entropy loss
copy "models\MRI\baseline\src\nnunet\training_docker\nnUNetTrainerV2_focalLoss.py" ".venv-nnunet\Lib\site-packages\nnunet\training\network_training\nnUNet_variants\loss_function\"
copy "models\MRI\baseline\src\nnunet\training_docker\nnUNetTrainerV2_Loss_FL_and_CE.py" ".venv-nnunet\Lib\site-packages\nnunet\training\network_training\"

# Verify installation
python -m uv run --python .venv-nnunet python -c "from nnunet.training.network_training.nnUNetTrainerV2_Loss_FL_and_CE import nnUNetTrainerV2_Loss_FL_and_CE_checkpoints; print('Custom trainers installed successfully')"

# Verify RTX 5090 compatibility
python -m uv run --python .venv-nnunet python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); test = torch.randn(1,3,4,4,4).cuda(); bn = torch.nn.BatchNorm3d(3).cuda(); out = bn(test); print('RTX 5090 WORKS!')"
```

## 3. CNN/ISUP Environment (.venv-cnn) - DONE

```powershell
# Install PyTorch (nightly for RTX 5090 sm_120 support)
python -m uv pip install --python .venv-cnn --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install dependencies
python -m uv pip install --python .venv-cnn numpy matplotlib pandas scikit-image nibabel SimpleITK tqdm tensorboard monai scikit-learn

# Verify RTX 5090 compatibility
python -m uv run --python .venv-cnn python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); test = torch.randn(1,3,4,4,4).cuda(); bn = torch.nn.BatchNorm3d(3).cuda(); out = bn(test); print('RTX 5090 WORKS!')"
```

## 4. nnDetection Environment (.venv-nndet)

**nnDetection does NOT support Windows!**


### WSL2 (Recommended for Windows)

nnDetection officially only supports Linux. Use Windows Subsystem for Linux 2:

```powershell
# In Windows PowerShell, install WSL2
wsl --install -d Ubuntu-22.04

# After installation, start WSL2
wsl

# Inside WSL2 terminal:
cd /mnt/c/workspace/uulm/PCa-classification

# Install uv in WSL2
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Create virtual environment (Python 3.10+ required for PyTorch nightly cu128)
# nnDetection supports Python 3.8+, so 3.10 works fine
uv venv .venv-nndet --python 3.10

# Install PyTorch (nightly for RTX 5090 sm_120 support)
# Note: PyTorch nightly cu128 requires Python 3.10+ (no wheels for 3.9)
uv pip install --python .venv-nndet --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify PyTorch installation and RTX 5090 support
.venv-nndet/bin/python -c 'import torch; print(f"PyTorch: {torch.__version__}"); print(f"CUDA: {torch.version.cuda}"); test = torch.randn(1,3,4,4,4).cuda(); bn = torch.nn.BatchNorm3d(3).cuda(); out = bn(test); print("RTX 5090 WORKS IN WSL!")'

# Clone nnDetection and install from source (specific commit used by picai_baseline)
git clone https://github.com/MIC-DKFZ/nnDetection.git /tmp/nnDetection
cd /tmp/nnDetection
git checkout 1044ace5340b2a07bf9f9d5f92681f712cc0d2b4

# WORKAROUND 1: Fix SimpleITK version conflict
# Old SimpleITK 1.2.0 doesn't build with modern CMake, use 2.x instead
sed -i 's/SimpleITK<2.1.0/SimpleITK>=2.0.0/' requirements.txt

# WORKAROUND 2: Fix PyTorch 2.x API compatibility in CUDA code
# Replace deprecated .type() with .scalar_type()
sed -i 's/dets_sorted\.type()/dets_sorted.scalar_type()/g' nndet/csrc/cuda/nms.cu

# WORKAROUND 3: Fix torch._six removal in PyTorch 2.x
# torch._six was removed in PyTorch 1.9+, replace with str
sed -i 's/from torch._six import string_classes/string_classes = str/' nndet/utils/tensor.py

# WORKAROUND 4: Fix matplotlib seaborn style name
# 'seaborn-deep' was renamed to 'seaborn-v0_8-deep' in matplotlib 3.6+
sed -i "s/plt.style.use('seaborn-deep')/plt.style.use('seaborn-v0_8-deep')/" nndet/utils/analysis.py

# Install dependencies
cd /mnt/c/workspace/uulm/PCa-classification
uv pip install --python .venv-nndet scikit-build cmake ninja
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
uv pip install --python .venv-nndet numpy pandas scikit-learn scipy nibabel batchgenerators scikit-image tqdm hydra-core matplotlib seaborn SimpleITK
uv pip install --python .venv-nndet hydra-core --upgrade
uv pip install --python .venv-nndet git+https://github.com/mibaumgartner/pytorch_model_summary.git

# Install CUDA 12.8 on WSL2 (must match PyTorch nightly cu128)
# Update package list
sudo apt-get update

# Install build essentials
sudo apt-get install -y build-essential

# Download CUDA 12.8 toolkit (matches PyTorch nightly cu128)
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run

# Install CUDA 12.8 (this will take a few minutes)
sudo sh cuda_12.8.0_570.86.10_linux.run --toolkit --silent --override

# Set CUDA 12.8 as active
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify CUDA version (should show 12.8)
nvcc --version

# Add to ~/.bashrc for persistence
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Install nnDetection with CUDA support
cd /mnt/c/workspace/uulm/PCa-classification
FORCE_CUDA=1 uv pip install --python .venv-nndet -e /tmp/nnDetection --no-build-isolation

# Install picai_prep for data conversion (nnUNet → nnDetection)
uv pip install --python .venv-nndet -e models/MRI/prep

# WORKAROUND 5: Add PyTorch libraries to LD_LIBRARY_PATH
# Required for nnDetection CUDA extensions to find libc10.so
export LD_LIBRARY_PATH=/mnt/c/workspace/uulm/PCa-classification/.venv-nndet/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/mnt/c/workspace/uulm/PCa-classification/.venv-nndet/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# Set required environment variables (add to ~/.bashrc for persistence)
export det_data=/mnt/c/workspace/uulm/PCa-classification/workdir/nnDet_data
export det_models=/mnt/c/workspace/uulm/PCa-classification/workdir/results/nnDet
export OMP_NUM_THREADS=1

echo 'export det_data=/mnt/c/workspace/uulm/PCa-classification/workdir/nnDet_data' >> ~/.bashrc
echo 'export det_models=/mnt/c/workspace/uulm/PCa-classification/workdir/results/nnDet' >> ~/.bashrc
echo 'export OMP_NUM_THREADS=1' >> ~/.bashrc

# Reload environment
source ~/.bashrc

# Verify nnDetection installation with CUDA extensions
.venv-nndet/bin/python -c 'import nndet; import nndet._C; print("nnDetection with CUDA extensions installed!")'

# Verify RTX 5090 support
.venv-nndet/bin/python -c 'import torch; test = torch.randn(1,3,4,4,4).cuda(); bn = torch.nn.BatchNorm3d(3).cuda(); out = bn(test); print("RTX 5090 WORKS IN WSL!")'

# Convert data from nnU-Net to nnDetection format
uv run --python .venv-nndet python -m picai_prep nnunet2nndet --input input/mri_data/Task2203 --output workdir/nnDet_data/Task2201_picai_baseline

# Run Tensorboard 
python -m tensorboard.main --logdir workdir/uda
