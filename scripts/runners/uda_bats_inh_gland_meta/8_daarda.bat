@echo off
:: DAARDA gland-prior+clinical UDA Experiments for merged public source -> UULM target
cd /d "%~dp0..\..\.."
echo Working directory: %CD%

set PYTHON=.venv-cnn\Scripts\python.exe
set SCRIPT=scripts\runners\1_cnn_uda_runner.py

set "BACKBONE=resnet10"
set "LR_SCHEDULER=--lr-scheduler inv"
set "SPLIT=--dataset-profile public_to_uulm --center-pairs RUMC+PCNN+ZGT_to_UULM"
set "DAARDA_ARGS=--daarda-divergence js_beta --daarda-relax 1.5 --daarda-grad-penalty 0.0"
set "UULM_META=--uulm-use-dual-metadata --uulm-label-file 0ii/man.xlsx --uulm-pet-label-file 0ii/pet.xlsx --uulm-pet-sheet-name Auswertung"
set "GLAND_ARGS=--model-variant prostate_clinical --prostate-prior-type whole_gland --prostate-prior-source bosma22b --prostate-prior-target pseudo --prostate-prior-target-dir 0ii/files/gland_masks --prostate-prior-cache-dir workdir/prostate_prior_cache"
set "COMMON=--da-method daarda --binary --backbone %BACKBONE% --plot-oracle --checkpoint-interval 2 --no-early-stopping --two-splits-source --target-cv-folds 3 --class-weights %LR_SCHEDULER% %SPLIT% %UULM_META% %GLAND_ARGS% --aug-all %DAARDA_ARGS%"

echo ========================================================
echo DAARDA GLAND-PRIOR+CLINICAL UDA Experiments (RUMC+PCNN+ZGT -> UULM)
echo ARDA settings: divergence=js_beta, relax=1.5, grad_penalty=0.0
echo ========================================================

%PYTHON% %SCRIPT% %COMMON% --dropout 0.3 --weight-decay 1e-4

echo.
echo All DAARDA gland-prior+clinical experiments completed!
