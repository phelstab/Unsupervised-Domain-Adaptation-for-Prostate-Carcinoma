@echo off
:: Run B1 and C1 with ALL four classifiers: LR, GP, Bayesian LR, SVM
:: Each classifier gets its own timestamped output directory.
::
:: Output: 8 timestamped directories under workdir\pet\
::   workdir\pet\YYYYMMDD_HHMMSS_b1_pet_metadata_source_val\           (LR)
::   workdir\pet\YYYYMMDD_HHMMSS_b1_pet_metadata_source_val_gp\        (GP)
::   workdir\pet\YYYYMMDD_HHMMSS_b1_pet_metadata_source_val_bayesian_lr\ (Bayesian LR)
::   workdir\pet\YYYYMMDD_HHMMSS_b1_pet_metadata_source_val_svm\       (SVM)
::   workdir\pet\YYYYMMDD_HHMMSS_c1_pet_radiomics_source_val_4feat\    (LR)
::   workdir\pet\YYYYMMDD_HHMMSS_c1_pet_radiomics_source_val_4feat_gp\ (GP)
::   etc.
cd /d "%~dp0..\..\.."
set PYTHON=.venv-cnn\Scripts\python.exe
for /f "tokens=1,* delims==" %%A in ('%PYTHON% scripts\select_pet_mri_run.py') do set "%%A=%%B"

echo ========================================================
echo PET SIDECAR: ALL CLASSIFIERS COMPARISON
echo Working directory: %CD%
echo Selected run: %RUN_DIR%
echo Selected algorithm: %ALGORITHM%
echo Selected DA weight: %DA_WEIGHT%
echo Classifiers: lr, gp, bayesian_lr, svm
echo Phases: B1 (metadata) + C1 (radiomics) x 4 classifiers = 8 runs
echo ========================================================
echo.

:: Use subroutine to avoid delayed expansion issues with set inside for loops
for %%C in (lr gp bayesian_lr svm) do call :run_classifier %%C

echo ========================================================
echo ALL CLASSIFIER COMPARISON RUNS COMPLETED!
echo Results in timestamped directories under workdir\pet\
echo ========================================================
pause
goto :eof

:run_classifier
set CLASSIFIER=%1
echo ----------------------------------------------------------
echo  Classifier: %CLASSIFIER%
echo ----------------------------------------------------------

echo [B1/%CLASSIFIER%] Running B1 with %CLASSIFIER% ...
call "%~dp0\b1_pet_metadata.bat"
echo.

echo [C1/%CLASSIFIER%] Running C1 with %CLASSIFIER% ...
call "%~dp0\c1_pet_radiomics.bat"
echo.
goto :eof
