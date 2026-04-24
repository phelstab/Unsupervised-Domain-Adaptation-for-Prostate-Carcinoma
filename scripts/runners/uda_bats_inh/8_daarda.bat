@echo off
:: DAARDA UDA Experiments for merged public source -> UULM target
cd /d "%~dp0..\..\.."
echo Working directory: %CD%

set PYTHON=.venv-cnn\Scripts\python.exe
set SCRIPT=scripts\runners\1_cnn_uda_runner.py

set "BACKBONE=resnet10"
set "LR_SCHEDULER=--lr-scheduler inv"
set "SPLIT=--dataset-profile public_to_uulm --center-pairs RUMC+PCNN+ZGT_to_UULM"
set "DAARDA_ARGS=--daarda-divergence js_beta --daarda-relax 1.5 --daarda-grad-penalty 0.0"
set "UULM_META=--uulm-use-dual-metadata --uulm-label-file 0ii/man.xlsx --uulm-pet-label-file 0ii/pet.xlsx --uulm-pet-sheet-name Auswertung"
set "COMMON=--da-method daarda --binary --backbone %BACKBONE% --plot-oracle --checkpoint-interval 2 --no-early-stopping --two-splits-source --target-cv-folds 3 --class-weights %LR_SCHEDULER% %SPLIT% %UULM_META% --aug-all %DAARDA_ARGS%"

echo ========================================================
echo DAARDA UDA Experiments (RUMC+PCNN+ZGT - UULM)
echo Total full-reg runs: 3 DA weights x 1 center pair = 3 runs
echo ARDA settings: divergence=js_beta, relax=1.5, grad_penalty=0.0
echo ========================================================

:: PHASE 1: Simple baseline
@REM %PYTHON% %SCRIPT% %COMMON% --dropout 0.0 --no-batchnorm --weight-decay 0.0

:: PHASE 2: Full regularization
%PYTHON% %SCRIPT% %COMMON% --dropout 0.3 --weight-decay 1e-4

echo.
echo All DAARDA experiments completed!
