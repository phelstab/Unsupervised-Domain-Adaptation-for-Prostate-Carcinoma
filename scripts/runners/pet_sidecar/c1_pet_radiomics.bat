@echo off
:: C1: PET Image Radiomics Fusion — Post-hoc classifier combining MRI score + radiomic features
:: Select MRI run automatically from completed A4-style UDA runs in runs\
:: using the retrospective source_val winner rule implemented in scripts\select_pet_mri_run.py
:: Default run excludes shape features to keep C1 focused on PET intensity/texture
cd /d "%~dp0..\..\.."
echo Working directory: %CD%

set PYTHON=.venv-cnn\Scripts\python.exe
set SCRIPT=scripts\runners\pet_sidecar\c1_pet_radiomics_runner.py

if not defined RUN_DIR (
  for /f "tokens=1,* delims==" %%A in ('%PYTHON% scripts\select_pet_mri_run.py') do set "%%A=%%B"
)

:: Set default classifier if not specified by caller
if not defined CLASSIFIER set CLASSIFIER=lr

:: Arguments:
::   --run-dir          Path to experiment run directory (averages MRI scores across 3 folds)
::   --da-weight        DA loss weight subdirectory (default: 0.9)
::   --validator        Epoch selection strategy: source_val (default), last
::   --C                Logistic regression regularization strength (default: 1.0)
::   --classifier       Classifier: lr (default), gp, bayesian_lr, svm
::   --n-features       Max features for RFE including MRI score (default: 4)
::   --suv-dir          Directory with SUV .mha files (default: workdir/pet/suv_volumes)
::   --gland-dir        Directory with gland mask .mha files (default: 0ii/files/gland_masks)
::   --pet-xlsx         Path to PET metadata spreadsheet (default: 0ii/pet.xlsx)
::   --output-dir       Output directory (default: workdir/pet)
::   --skip-extraction  Skip radiomics extraction, load from the mode-specific cache
::   --include-shape-features  Add ROI shape features from the resampled gland mask
::   --permutation-test N  Run permutation test with N shuffles (0 = skip)

echo ========================================================
echo C1: PET Image Radiomics Fusion (MRI + PET radiomic features)
echo Run dir: %RUN_DIR%
echo Algorithm: %ALGORITHM%
echo DA weight: %DA_WEIGHT%
echo Validator: source_val
echo Classifier: %CLASSIFIER%
echo Shape features: NO (default clean PET radiomics run)
echo Skip extraction: NO (safe mode-specific cache handling)
echo Permutation test: 1000 shuffles
echo ========================================================

%PYTHON% %SCRIPT% --run-dir "%RUN_DIR%" --da-weight %DA_WEIGHT% --validator source_val --classifier %CLASSIFIER% --permutation-test 1000

echo.
echo C1 completed! Results in timestamped directory under workdir\pet\
