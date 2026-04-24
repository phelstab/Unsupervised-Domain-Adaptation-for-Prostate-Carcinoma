@echo off
REM PI-CAI Baseline Directory Setup Script for Windows
REM Native Windows batch script

echo Setting up PI-CAI baseline directory structure...

REM Get the current directory
set "ROOT_DIR=%~dp0"
set "ROOT_DIR=%ROOT_DIR:~0,-1%"

echo Root directory: %ROOT_DIR%

echo Creating main directories...

REM Create main directories
if not exist "%ROOT_DIR%\workdir" mkdir "%ROOT_DIR%\workdir"
if not exist "%ROOT_DIR%\workdir\nnUNet_raw_data" mkdir "%ROOT_DIR%\workdir\nnUNet_raw_data"
if not exist "%ROOT_DIR%\workdir\nnDet_raw_data" mkdir "%ROOT_DIR%\workdir\nnDet_raw_data"
if not exist "%ROOT_DIR%\workdir\results" mkdir "%ROOT_DIR%\workdir\results"
if not exist "%ROOT_DIR%\workdir\results\UNet" mkdir "%ROOT_DIR%\workdir\results\UNet"
if not exist "%ROOT_DIR%\workdir\results\UNet\weights" mkdir "%ROOT_DIR%\workdir\results\UNet\weights"
if not exist "%ROOT_DIR%\workdir\results\UNet\overviews" mkdir "%ROOT_DIR%\workdir\results\UNet\overviews"
if not exist "%ROOT_DIR%\workdir\results\nnUNet" mkdir "%ROOT_DIR%\workdir\results\nnUNet"
if not exist "%ROOT_DIR%\workdir\results\nnDet" mkdir "%ROOT_DIR%\workdir\results\nnDet"
if not exist "%ROOT_DIR%\workdir\splits" mkdir "%ROOT_DIR%\workdir\splits"
if not exist "%ROOT_DIR%\input" mkdir "%ROOT_DIR%\input"
if not exist "%ROOT_DIR%\input\images" mkdir "%ROOT_DIR%\input\images"
if not exist "%ROOT_DIR%\output" mkdir "%ROOT_DIR%\output"
if not exist "%ROOT_DIR%\logs" mkdir "%ROOT_DIR%\logs"

echo Directory structure created successfully!
echo.
echo Created directories:
echo    workdir\                    - Working directory for processing
echo    workdir\nnUNet_raw_data\    - nnU-Net preprocessed data
echo    workdir\nnDet_raw_data\     - nnDetection preprocessed data
echo    workdir\results\            - Training results and models
echo    workdir\splits\             - Cross-validation splits
echo    input\                      - Input data directory
echo    input\images\               - PI-CAI dataset images
echo    output\                     - Final outputs
echo    logs\                       - Training and processing logs
echo.
echo Next steps:
echo    1. Download PI-CAI dataset to input\images\
echo    2. Ensure picai_labels is available in Datasets\MRI\PI-CAI_Challenge_2024\
echo    3. Run data preparation scripts
echo.
echo Usage: setup_directories.bat

pause
