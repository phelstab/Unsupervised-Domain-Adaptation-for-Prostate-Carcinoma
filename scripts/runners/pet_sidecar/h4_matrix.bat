@echo off
:: Thesis Experiment III (H4): per-validator B1 + C1 PET fusion.
::
:: Runs the three pipeline stages end-to-end:
::   Stage 1. Extract radiomic features (C1 cache) with shape + binWidth=0.05
::            (Zamboglou 2019 / Solari 2022 defaults)
::   Stage 2. For every validator, pick one A4 checkpoint per fold and cache
::            the 25-patient PET-subset OOF MRI score vector.
::   Stage 3. Run B1 + C1 LOOCV once per validator and emit the 6x2 matrix
::            plus LaTeX table ready for results.tex Section 5.3.
::
:: Usage:
::   scripts\runners\pet_sidecar\h4_matrix.bat
::   scripts\runners\pet_sidecar\h4_matrix.bat 1000   (with permutation test)
::
:: Output goes to workdir\pet\YYYYMMDD_HHMMSS_h4_matrix[_permN]\

setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0..\..\.."
set PYTHON=.venv-cnn\Scripts\python.exe
set PERM=%1
if "%PERM%"=="" set PERM=0

echo ========================================================
echo H4 MATRIX RUNNER (Experiment III: B1 + C1 x 6 validators)
echo ========================================================

echo.
echo [1/3] Ensuring radiomics cache (shape ON, binWidth=0.05)...
%PYTHON% scripts\runners\pet_sidecar\c1_pet_radiomics_runner.py --dry-run ^
    --include-shape-features --bin-width 0.05
if errorlevel 1 goto :fail

echo.
echo [2/3] Per-validator A4 pool score extraction...
%PYTHON% scripts\a4_pool_score_extraction.py
if errorlevel 1 goto :fail

echo.
echo [3/3] H4 matrix runner...
if "%PERM%"=="0" (
    %PYTHON% scripts\runners\pet_sidecar\h4_matrix_runner.py ^
        --radiomics-csv workdir\pet\c1_radiomics_features_with_shape_bw0p05.csv
) else (
    %PYTHON% scripts\runners\pet_sidecar\h4_matrix_runner.py ^
        --radiomics-csv workdir\pet\c1_radiomics_features_with_shape_bw0p05.csv ^
        --permutation-test %PERM%
)
if errorlevel 1 goto :fail

echo.
echo ========================================================
echo H4 MATRIX COMPLETE -- see workdir\pet\*_h4_matrix*\
echo ========================================================
goto :eof

:fail
echo FAILED at stage; see log files above.
exit /b 1
