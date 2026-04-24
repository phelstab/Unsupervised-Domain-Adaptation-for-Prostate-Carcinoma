@echo off
:: H4 matrix with metadata-augmented C1.
::
:: Runs the full pipeline (Stage 1: radiomics, Stage 2: validator caches,
:: Stage 3: matrix) and then also runs a second Stage 3 pass with the
:: --include-metadata flag so C1 is evaluated with the three B1 spreadsheet
:: features added to its RFE pool.
::
:: This produces TWO timestamped run dirs:
::   workdir\pet\YYYYMMDD_HHMMSS_h4_matrix\            (thesis-canonical C1)
::   workdir\pet\YYYYMMDD_HHMMSS_h4_matrix_with_meta\  (diagnostic C1+meta)
::
:: Total wall time ~15-20 min (Stage 1+2 once, Stage 3 twice).
::
:: Usage:
::   scripts\runners\pet_sidecar\h4_matrix_with_meta.bat

setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0..\..\.."
set PYTHON=.venv-cnn\Scripts\python.exe

echo ========================================================
echo H4 MATRIX (canonical + metadata-augmented)
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
echo [3/4] H4 matrix (canonical C1: radiomics only)...
%PYTHON% scripts\runners\pet_sidecar\h4_matrix_runner.py ^
    --radiomics-csv workdir\pet\c1_radiomics_features_with_shape_bw0p05.csv
if errorlevel 1 goto :fail

echo.
echo [4/4] H4 matrix (diagnostic: C1 pool augmented with B1 metadata)...
%PYTHON% scripts\runners\pet_sidecar\h4_matrix_runner.py ^
    --radiomics-csv workdir\pet\c1_radiomics_features_with_shape_bw0p05.csv ^
    --include-metadata
if errorlevel 1 goto :fail

echo.
echo ========================================================
echo BOTH MATRICES COMPLETE
echo Canonical:  workdir\pet\*_h4_matrix\h4_matrix.csv
echo Diagnostic: workdir\pet\*_h4_matrix_with_meta\h4_matrix.csv
echo ========================================================
goto :eof

:fail
echo FAILED at stage; see log files above.
exit /b 1
