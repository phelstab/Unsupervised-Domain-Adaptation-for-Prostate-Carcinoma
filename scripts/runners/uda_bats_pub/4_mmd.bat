@echo off
:: MMD UDA Experiments for public dataset splits (2-split mode)
cd /d "%~dp0..\..\.."
echo Working directory: %CD%

set PYTHON=.venv-cnn\Scripts\python.exe
set SCRIPT=scripts\runners\1_cnn_uda_runner.py

set "BACKBONE=resnet10"
set "LR_SCHEDULER=--lr-scheduler inv"
set "COMMON=--da-method mmd --binary --backbone %BACKBONE% --plot-oracle --checkpoint-interval 2 --no-early-stopping --two-splits-source --two-splits-target --target-cv-folds 3 --class-weights %LR_SCHEDULER% --aug-all"

echo ========================================================
echo MMD UDA Experiments (Public Splits 2)
echo Split: RUMC_to_PCNN, RUMC_to_ZGT (2-split mode)
echo Full regularization with class weights
echo Total full-reg runs: 3 DA weights x 2 center pairs = 6 runs
echo ========================================================

:: PHASE 1: Simple baseline
@REM %PYTHON% %SCRIPT% %COMMON% --dropout 0.0 --no-batchnorm --weight-decay 0.0

:: PHASE 2: Full regularization
%PYTHON% %SCRIPT% %COMMON% --dropout 0.3 --weight-decay 1e-4

echo.
echo All MMD experiments completed!
