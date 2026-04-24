@echo off
:: Main runner: Execute all gland-prior+clinical UDA algorithm experiments for merged public -> UULM
cd /d "%~dp0..\..\.."
echo ========================================================
echo MAIN UDA GLAND-PRIOR+CLINICAL RUNNER
echo Working directory: %CD%
echo Split: RUMC+PCNN+ZGT -> UULM
echo Variant: prostate_clinical (whole_gland + clinical metadata)
echo Algorithms: MCD, DANN, MMD, HYBRID, MCC, BNM, DAARDA
echo ========================================================
echo.

echo [2/7] Running DANN gland-prior+clinical experiments...
call "%~dp0\3_dann.bat"

echo [3/7] Running MMD gland-prior+clinical experiments...
call "%~dp0\4_mmd.bat"

echo [1/7] Running MCD gland-prior+clinical experiments...
call "%~dp0\2_mcd.bat"

echo [4/7] Running HYBRID gland-prior+clinical experiments...
call "%~dp0\5_hybrid.bat"

echo [5/7] Running MCC gland-prior+clinical experiments...
call "%~dp0\6_mcc.bat"

echo [6/7] Running BNM gland-prior+clinical experiments...
call "%~dp0\7_bnm.bat"

echo [7/7] Running DAARDA gland-prior+clinical experiments...
call "%~dp0\8_daarda.bat"

echo.
echo ========================================================
echo ALL GLAND-PRIOR+CLINICAL EXPERIMENTS COMPLETED!
echo ========================================================
pause
