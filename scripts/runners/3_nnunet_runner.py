# UV Virtual Environment: .venv-nnunet
# Setup: uv venv .venv-nnunet
# Install PyTorch: uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# Install dependencies: uv pip install nnunet==1.7.0 batchgenerators scikit-image scipy nibabel SimpleITK scikit-learn tqdm
# Install PI-CAI deps: uv pip install picai-baseline picai-eval report-guided-annotation
# Data location: input/mri_data/Task2203/ (imagesTr/, labelsTr/)
# Note: nnUNet expects data in workdir/nnUNet_raw_data/Task2203/ - create symlink or copy from input/mri_data/Task2203/
# Run: python -m uv run --python .venv-nnunet python scripts\runners\3_nnunet_runner.py --inputdir input --workdir workdir --task Task2203 --folds 0

import os
import sys
import argparse
import shutil
import subprocess
from pathlib import Path

def _which(cmd):
    return shutil.which(cmd) is not None

def _run(cmd):
    print('>', ' '.join(map(str, cmd)), flush=True)
    subprocess.check_call([str(c) for c in cmd])

def _nnunet_cmd():
    if _which('nnunet'):
        return ['nnunet']
    return None

def plan_train(task, workdir, trainer, custom_split, fold):
    wrapper = _nnunet_cmd()
    if wrapper:
        cmd = wrapper + ['plan_train', task, workdir, '--trainer', trainer, '--fold', str(fold)]
        if custom_split:
            cmd += ['--custom_split', custom_split]
        _run(cmd)
    else:
        w = Path(workdir).absolute()
        os.environ['nnUNet_raw_data_base'] = str(w)
        os.environ['nnUNet_preprocessed'] = str(w / 'nnUNet_preprocessed')
        os.environ['RESULTS_FOLDER'] = str(w / 'results' / 'nnUNet')
        
        task_id = task.replace('Task', '').split('_')[0]
        
        _run([sys.executable, '-m', 'nnunet.experiment_planning.nnUNet_plan_and_preprocess', 
              '-t', task_id, '--verify_dataset_integrity'])
        
        _run([sys.executable, '-m', 'nnunet.run.run_training', 
              '3d_fullres', trainer, task_id, str(fold), '--npz'])

def predict(task, workdir, trainer, fold, input_dir, output_dir):
    wrapper = _nnunet_cmd()
    if wrapper:
        cmd = wrapper + ['predict', task, '--trainer', trainer, '--fold', str(fold), 
                        '--checkpoint', 'model_best', '--results', str(Path(workdir) / 'results'), 
                        '--input', input_dir, '--output', output_dir, '--store_probability_maps']
        _run(cmd)
    else:
        task_id = task.replace('Task', '').split('_')[0]
        _run([sys.executable, '-m', 'nnunet.inference.predict', 
              '-i', input_dir, '-o', output_dir, '-t', task_id, 
              '-m', '3d_fullres', '-f', str(fold), '-chk', 'model_best'])

def main():
    parser = argparse.ArgumentParser(description='nnU-Net Runner (Windows, No Docker)')
    
    parser.add_argument('--inputdir', type=str, required=True,
                       help='Input data directory (e.g., input)')
    parser.add_argument('--workdir', type=str, required=True,
                       help='Working directory (e.g., workdir)')
    parser.add_argument('--task', type=str, default='Task2201_picai_baseline',
                       help='Task name')
    parser.add_argument('--trainer', type=str, default='nnUNetTrainerV2_Loss_FL_and_CE_checkpoints',
                       help='Trainer class name')
    parser.add_argument('--folds', nargs='+', default=['0', '1', '2', '3', '4'],
                       help='Folds to train')
    parser.add_argument('--custom_split', type=str, default='',
                       help='Optional path to JSON splits file')
    parser.add_argument('--run_eval', action='store_true',
                       help='Run evaluation (requires picai-eval, report-guided-annotation)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training phase')
    parser.add_argument('--skip_inference', action='store_true',
                       help='Skip inference phase')
    
    args = parser.parse_args()
    
    os.environ['inputdir'] = os.path.abspath(args.inputdir)
    os.environ['workdir'] = os.path.abspath(args.workdir)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    
    workdir = Path(args.workdir)
    # nnUNet expects data in nnUNet_raw_data - if not found, check input/mri_data
    task_dir = workdir / 'nnUNet_raw_data' / args.task
    if not task_dir.exists():
        # Try alternate location
        task_dir_alt = Path('input') / 'mri_data' / args.task.replace('_picai_baseline', '')
        if task_dir_alt.exists():
            print(f'Data found at {task_dir_alt}, copying to {task_dir}')
            task_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(task_dir_alt), str(task_dir), dirs_exist_ok=True)
            print(f'Data copied successfully')
    
    imagesTr = task_dir / 'imagesTr'
    preds_root = task_dir
    
    if not args.skip_training:
        print('=== Training ===')
        for f in args.folds:
            model_final = workdir / f'results/nnUNet/3d_fullres/{args.task}/{args.trainer}__nnUNetPlansv2.1/fold_{f}/model_final.model'
            if model_final.exists():
                print(f'Training already finished for fold {f}, skipping')
                continue
            plan_train(args.task, str(workdir), args.trainer, args.custom_split or '', f)
    
    if not args.skip_inference:
        print('=== Inference ===')
        for f in args.folds:
            out_dir = preds_root / f'predictions_fold_{f}'
            if out_dir.exists():
                print(f'Inference already finished for fold {f}, skipping')
                continue
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            predict(args.task, str(workdir), args.trainer, f, str(imagesTr), str(out_dir))
    
    if args.run_eval:
        print('=== Evaluation ===')
        try:
            from picai_eval import evaluate_folder
            from picai_baseline.splits.picai_nnunet import train_splits, valid_splits
            from report_guided_annotation import extract_lesion_candidates
            
            for f in map(int, args.folds):
                pred_dir = preds_root / f'predictions_fold_{f}'
                m_path = pred_dir / 'metrics.json'
                if not m_path.exists():
                    metrics = evaluate_folder(
                        y_det_dir=str(pred_dir),
                        y_true_dir=str(task_dir / 'labelsTr'),
                        subject_list=valid_splits[f],
                        y_det_postprocess_func=lambda p: extract_lesion_candidates(p)[0],
                    )
                    metrics.save(str(m_path))
                
                mt_path = pred_dir / 'metrics-train.json'
                if not mt_path.exists():
                    metrics = evaluate_folder(
                        y_det_dir=str(pred_dir),
                        y_true_dir=str(task_dir / 'labelsTr'),
                        subject_list=train_splits[f],
                        y_det_postprocess_func=lambda p: extract_lesion_candidates(p)[0],
                    )
                    metrics.save(str(mt_path))
        except Exception as e:
            print(f'Evaluation skipped (missing deps or error): {e}')
    
    print('Done!')

if __name__ == '__main__':
    main()
