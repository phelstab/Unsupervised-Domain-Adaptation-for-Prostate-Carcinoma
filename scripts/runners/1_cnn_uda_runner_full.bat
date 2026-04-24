@echo off
:: Navigate to the workspace root (2 levels up from scripts\runners)
cd /d "%~dp0..\.."
echo Working directory: %CD%

set PYTHON=.venv-cnn\Scripts\python.exe
set SCRIPT=scripts\runners\1_cnn_uda_runner.py

:: "Start simple" config:
:: --dropout 0.0 = no dropout (default)
:: --no-batchnorm = disable batch normalization
:: --weight-decay 0.0 = no weight decay (default)
:: --lr-scheduler none = no LR scheduler (default)
:: --checkpoint-interval 2 = save checkpoints every 2 epochs (50 checkpoints per 100 epochs)
:: --no-early-stopping = run full 100 epochs without early stopping
:: --backbone = model backbone (simple, resnet10, resnet18, resnet50)
set SIMPLE_FLAGS=--dropout 0.0 --no-batchnorm --weight-decay 0.0 --lr-scheduler none --checkpoint-interval 2 --no-early-stopping

:: "Advanced" config (enable regularization techniques):
:: Uncomment and use ADVANCED_FLAGS instead of SIMPLE_FLAGS to enable:
:: --dropout 0.5 = 50% dropout
:: --weight-decay 1e-4 = weight decay regularization
:: --lr-scheduler step = StepLR with step_size=30, gamma=0.1
:: --lr-scheduler cosine = CosineAnnealingLR
:: set ADVANCED_FLAGS=--dropout 0.5 --weight-decay 1e-4 --lr-scheduler step --lr-step-size 30 --lr-gamma 0.1 --checkpoint-interval 2 --no-early-stopping

:: Backbone selection (resnet10/resnet18/resnet34/resnet50 = 3D ResNet variants)
:: ResNet10 recommended for small datasets, ResNet34 is a good middle ground
set BACKBONE=resnet10

:: Oracle plotting (set to --plot-oracle to enable, or leave empty to disable)
set PLOT_ORACLE=--plot-oracle

:: Data split mode (default: 3-split = 70/15/15 train/val/test)
:: 2-split = 85/15 train/eval (use when data is limited - merges val+test)
:: Uncomment to enable 2-split mode:
:: set SPLIT_FLAGS=--two-splits-source --two-splits-target
set SPLIT_FLAGS=

echo ========================================================
echo Configuration: Starting simple (no dropout, no batchnorm, no weight decay, no LR scheduler)
echo Backbone: %BACKBONE%
echo Checkpoint interval: every 2 epochs, No early stopping
echo ========================================================

@REM echo ========================================================
@REM echo 5. Running DANN UDA (Domain Adversarial Neural Network)
@REM echo    Adversarial domain adaptation with gradient reversal
@REM echo ========================================================
@REM %PYTHON% %SCRIPT% --da-method dann --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS%

@REM echo ========================================================
@REM echo 1. Running Lower Bound (Source Only, source-based selection)
@REM echo ========================================================
@REM @REM set BACKBONE=simple
@REM @REM %PYTHON% %SCRIPT% --lower-bound --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS%
@REM set BACKBONE=resnet10
@REM %PYTHON% %SCRIPT% --lower-bound --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS%
@REM set BACKBONE=resnet50
@REM %PYTHON% %SCRIPT% --lower-bound --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS%

@REM echo ========================================================
@REM echo 2. Running Upper Bound (Source + Target Supervised)
@REM echo ========================================================
@REM @REM set BACKBONE=simple
@REM @REM %PYTHON% %SCRIPT% --upper-bound --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS%
@REM set BACKBONE=resnet10
@REM %PYTHON% %SCRIPT% --upper-bound --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS%
@REM set BACKBONE=resnet50
@REM %PYTHON% %SCRIPT% --upper-bound --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS%

echo ========================================================
echo 3. UDA: Hybrid (CORAL + EntropyMin) (source_val checkpoint selection)
echo    This is the realistic UDA setting per advisor recommendation
echo ========================================================
@REM set BACKBONE=simple
@REM %PYTHON% %SCRIPT% --da-method hybrid --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS%
set BACKBONE=resnet10
%PYTHON% %SCRIPT% --da-method hybrid --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS%
set BACKBONE=resnet50
%PYTHON% %SCRIPT% --da-method hybrid --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS%

echo ========================================================
echo 4. UDA: MMD (Maximum Mean Discrepancy)
echo    Feature distance loss alternative to CORAL
echo ========================================================
@REM set BACKBONE=simple
@REM %PYTHON% %SCRIPT% --da-method mmd --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS%
set BACKBONE=resnet10
%PYTHON% %SCRIPT% --da-method mmd --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS%
set BACKBONE=resnet50
%PYTHON% %SCRIPT% --da-method mmd --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS%

echo ========================================================
echo 5. UDA: DANN (Domain Adversarial Neural Network)
echo    Adversarial domain adaptation with gradient reversal
echo ========================================================
@REM set BACKBONE=simple
@REM %PYTHON% %SCRIPT% --da-method dann --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS%
set BACKBONE=resnet10
%PYTHON% %SCRIPT% --da-method dann --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS%
set BACKBONE=resnet50
%PYTHON% %SCRIPT% --da-method dann --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS%

@REM echo ========================================================
@REM echo 6. Running Hybrid UDA - ORACLE UPPER BOUND (target_val selection)
@REM echo    For comparison only - uses target labels (not realistic UDA)
@REM echo ========================================================
@REM @REM set BACKBONE=simple
@REM @REM %PYTHON% %SCRIPT% --da-method hybrid --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS% --checkpoint-validator target_val
@REM set BACKBONE=resnet10
@REM %PYTHON% %SCRIPT% --da-method hybrid --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS% --checkpoint-validator target_val
@REM set BACKBONE=resnet50
@REM %PYTHON% %SCRIPT% --da-method hybrid --binary %SIMPLE_FLAGS% --backbone %BACKBONE% %PLOT_ORACLE% %SPLIT_FLAGS% --checkpoint-validator target_val

echo.
echo All experiments completed!
pause