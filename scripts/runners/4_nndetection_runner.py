# UV Virtual Environment: .venv-nndet (WSL2 REQUIRED - nnDetection does NOT support Windows)
# IMPORTANT: nnDetection officially does NOT support Windows. Use WSL2 (Windows Subsystem for Linux)
#
# WSL2 Setup:
# 1. Install WSL2: wsl --install -d Ubuntu-22.04
# 2. Inside WSL2, navigate to project: cd /mnt/c/workspace/uulm/PCa-classification
# 3. Install uv: pip install uv
# 4. Create venv: uv venv .venv-nndet
# 5. Install PyTorch: uv pip install --python .venv-nndet --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# Data location: input/mri_data/Task2203/ (imagesTr/, labelsTr/)
# 6. Clone and install nnDetection from source:
#    git clone https://github.com/MIC-DKFZ/nnDetection.git /tmp/nnDetection
#    cd /tmp/nnDetection && git checkout 1044ace5340b2a07bf9f9d5f92681f712cc0d2b4
#    uv pip install --python /mnt/c/workspace/uulm/PCa-classification/.venv-nndet -r requirements.txt
#    uv pip install --python /mnt/c/workspace/uulm/PCa-classification/.venv-nndet hydra-core --upgrade
#    uv pip install --python /mnt/c/workspace/uulm/PCa-classification/.venv-nndet git+https://github.com/mibaumgartner/pytorch_model_summary.git
#    FORCE_CUDA=1 uv pip install --python /mnt/c/workspace/uulm/PCa-classification/.venv-nndet -e .
# 7. Run: uv run --python .venv-nndet python scripts/runners/4_nndetection_runner.py --workdir workdir --fold 0
#
# Alternative: Use Docker (see models/MRI/baseline/src/nndetection/training_docker/)

import os
import sys
import argparse
import types
from pathlib import Path

try:
    import shutil_sol
except Exception:
    import shutil
    m = types.ModuleType('shutil_sol')
    
    def copyfile(src, dst):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    
    def copytree(src, dst):
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(src, dst, dirs_exist_ok=True)
            return
        except TypeError:
            pass
        
        src = Path(src)
        for p in src.rglob('*'):
            rel = p.relative_to(src)
            q = dst / rel
            if p.is_dir():
                q.mkdir(parents=True, exist_ok=True)
            else:
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, q)
    
    m.copyfile = copyfile
    m.copytree = copytree
    sys.modules['shutil_sol'] = m

def run_nndet_training(task, workdir, fold, custom_split=''):
    import subprocess
    
    workdir = Path(workdir).absolute()
    
    os.environ['det_data'] = str(workdir / 'nnDet_data')
    os.environ['det_models'] = str(workdir / 'results' / 'nnDet')
    
    # nnDetection scripts are standalone files, not a module
    train_script = '/tmp/nnDetection/scripts/train.py'
    
    # nnDetection uses Hydra config system
    # Format: train.py <task> -o key1=val1 key2=val2
    cmd = [
        sys.executable, str(train_script),
        task,
        '-o', f'exp.fold={fold}',
    ]
    
    if custom_split:
        cmd.extend(['-o', f'exp.custom_split={custom_split}'])
    
    print('Running nnDetection training:', ' '.join(cmd))
    subprocess.check_call(cmd)

def run_nndet_consolidate(task, model, workdir, custom_split=''):
    import subprocess
    
    workdir = Path(workdir).absolute()
    
    os.environ['det_models'] = str(workdir / 'results' / 'nnDet')
    
    consolidate_script = '/tmp/nnDetection/scripts/consolidate.py'
    
    cmd = [
        sys.executable, str(consolidate_script),
        task,
        model,
        '-r', os.environ['det_models'],
    ]
    
    if custom_split:
        cmd += ['--custom_split', custom_split]
    
    print('Running nnDetection consolidation:', ' '.join(cmd))
    subprocess.check_call(cmd)

def run_nndet_predict(task, model, workdir, input_dir, output_dir, fold):
    import subprocess
    
    workdir = Path(workdir).absolute()
    
    os.environ['det_models'] = str(workdir / 'results' / 'nnDet')
    
    predict_script = '/tmp/nnDetection/scripts/predict.py'
    
    cmd = [
        sys.executable, str(predict_script),
        task,
        model,
        '-r', os.environ['det_models'],
        '-i', input_dir,
        '-o', output_dir,
        '-f', str(fold),
        '--resume',
    ]
    
    print('Running nnDetection prediction:', ' '.join(cmd))
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser(description='nnDetection Runner (Windows, No Docker)')
    
    parser.add_argument('--workdir', type=str, required=True,
                       help='Working directory (e.g., workdir)')
    parser.add_argument('--task', type=str, default='Task2201_picai_baseline',
                       help='Task name')
    parser.add_argument('--model', type=str, default='RetinaUNetV001_D3V001_3d',
                       help='Model name')
    parser.add_argument('--fold', type=str, default='0',
                       help='Fold number')
    parser.add_argument('--custom_split', type=str, default='',
                       help='Optional path to custom split file')
    parser.add_argument('--prep_only', action='store_true',
                       help='Only run preprocessing/planning')
    parser.add_argument('--skip_consolidate', action='store_true',
                       help='Skip consolidation step')
    parser.add_argument('--predict_input', type=str, default='',
                       help='Input directory for prediction')
    parser.add_argument('--predict_output', type=str, default='',
                       help='Output directory for prediction')
    
    args = parser.parse_args()
    
    workdir = Path(args.workdir)
    
    os.environ.setdefault('det_data', str(workdir / 'nnDet_data'))
    os.environ.setdefault('det_models', str(workdir / 'results' / 'nnDet'))
    os.environ.setdefault('det_num_threads', '6')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    
    print(f'det_data: {os.environ["det_data"]}')
    print(f'det_models: {os.environ["det_models"]}')
    
    if not args.prep_only:
        print('=== Training ===')
        run_nndet_training(args.task, args.workdir, args.fold, args.custom_split)
    
    if not args.skip_consolidate:
        print('=== Consolidation ===')
        run_nndet_consolidate(args.task, args.model, args.workdir, args.custom_split)
    
    if args.predict_input:
        if not args.predict_output:
            raise SystemExit('--predict_output required when --predict_input is set')
        
        print('=== Prediction ===')
        run_nndet_predict(args.task, args.model, args.workdir, 
                         args.predict_input, args.predict_output, args.fold)
    
    print('Done!')

if __name__ == '__main__':
    main()
