@echo off
:: Main runner: Execute B1 (PET metadata sidecar) and C1 (PET radiomics fusion)
:: Both are post-hoc classifiers on top of the selected A4 MRI UDA backbone
::
:: Prerequisites for C1 (radiomics):
::   1. .venv-cnn\Scripts\python.exe scripts\pet_data_explorer.py
::   2. .venv-cnn\Scripts\python.exe scripts\pet_dicom_to_suv.py
::   These generate workdir\pet\pet_series_catalog.json and
::   workdir\pet\suv_volumes\*.mha, respectively.
::
:: Output: Timestamped directories under workdir\pet\
::   workdir\pet\YYYYMMDD_HHMMSS_b1_pet_metadata_<config>\
::   workdir\pet\YYYYMMDD_HHMMSS_c1_pet_radiomics_<config>\
cd /d "%~dp0..\..\.."
set PYTHON=.venv-cnn\Scripts\python.exe
for /f "tokens=1,* delims==" %%A in ('%PYTHON% scripts\select_pet_mri_run.py') do set "%%A=%%B"

echo ========================================================
echo MAIN PET SIDECAR RUNNER
echo Working directory: %CD%
echo Phase B1: MRI score + PET spreadsheet metadata (LOOCV LR)
echo Phase C1: MRI score + PET image radiomics (LOOCV LR+RFE, no shape by default)
echo Base model: selected A4 MRI UDA run, source_val validator
echo Selected run: %RUN_DIR%
echo Selected algorithm: %ALGORITHM%
echo Selected DA weight: %DA_WEIGHT%
echo Output: Timestamped directories in workdir\pet\
echo ========================================================
echo.

echo [1/2] Running B1: PET Metadata Sidecar...
call "%~dp0\b1_pet_metadata.bat"

echo.

echo [2/2] Running C1: PET Radiomics Fusion...
call "%~dp0\c1_pet_radiomics.bat"

echo.
echo ========================================================
echo ALL PET SIDECAR EXPERIMENTS COMPLETED!
echo Results in timestamped directories under workdir\pet\
echo ========================================================
pause
