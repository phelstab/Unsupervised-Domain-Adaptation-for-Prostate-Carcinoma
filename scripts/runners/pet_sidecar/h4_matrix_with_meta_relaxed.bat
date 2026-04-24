@echo off
:: H4 matrix with metadata-augmented C1 + relaxed label (ISUP >= 1).
::
:: Identical to h4_matrix_with_meta.bat but adds --relaxed-label so
:: Gleason 6 (GG1) patients count as positive (N+ = 8 instead of 3).
:: Stages 1+2 are shared with the canonical run; only Stage 3 differs.
::
:: This produces TWO timestamped run dirs:
::   workdir\pet\YYYYMMDD_HHMMSS_h4_matrix_relaxed_label\            (canonical C1)
::   workdir\pet\YYYYMMDD_HHMMSS_h4_matrix_with_meta_relaxed_label\  (diagnostic C1+meta)
::
:: Usage:
::   scripts\runners\pet_sidecar\h4_matrix_with_meta_relaxed.bat

setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0..\..\.."
set PYTHON=.venv-cnn\Scripts\python.exe

echo ========================================================
echo H4 MATRIX (relaxed label: ISUP ge 1, canonical + meta)
echo ========================================================

echo.
echo [1/4] Radiomics cache (shape ON, binWidth=0.05)...
%PYTHON% scripts\runners\pet_sidecar\c1_pet_radiomics_runner.py --dry-run ^
    --include-shape-features --bin-width 0.05
if errorlevel 1 goto :fail

echo.
echo [2/4] Per-validator A4 pool score extraction...
%PYTHON% scripts\a4_pool_score_extraction.py
if errorlevel 1 goto :fail

echo.
echo [3/4] H4 matrix (canonical C1, relaxed label)...
%PYTHON% scripts\runners\pet_sidecar\h4_matrix_runner.py ^
    --radiomics-csv workdir\pet\c1_radiomics_features_with_shape_bw0p05.csv ^
    --relaxed-label
if errorlevel 1 goto :fail

echo.
echo [4/4] H4 matrix (diagnostic C1+meta, relaxed label)...
%PYTHON% scripts\runners\pet_sidecar\h4_matrix_runner.py ^
    --radiomics-csv workdir\pet\c1_radiomics_features_with_shape_bw0p05.csv ^
    --include-metadata --relaxed-label
if errorlevel 1 goto :fail

echo.
echo ========================================================
echo BOTH MATRICES COMPLETE (relaxed label)
echo Canonical:  workdir\pet\*_h4_matrix_relaxed_label\h4_matrix.csv
echo Diagnostic: workdir\pet\*_h4_matrix_with_meta_relaxed_label\h4_matrix.csv
echo ========================================================
goto :eof

:fail
echo FAILED at stage; see log files above.
exit /b 1
