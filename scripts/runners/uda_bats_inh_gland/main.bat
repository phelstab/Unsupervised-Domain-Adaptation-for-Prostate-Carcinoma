@echo off
:: Main runner: Execute all gland-prior UDA algorithm experiments for merged public -> UULM
cd /d "%~dp0..\..\.."
echo ========================================================
echo MAIN UDA GLAND-PRIOR RUNNER
echo Working directory: %CD%
echo Split: RUMC+PCNN+ZGT -> UULM
echo Variant: prostate_prior (whole_gland)
echo Algorithms: MCD, DANN, MMD, HYBRID, MCC, BNM, DAARDA
echo ========================================================
echo.

echo [2/7] Running DANN gland-prior experiments...
call "%~dp0\3_dann.bat"

echo [3/7] Running MMD gland-prior experiments...
call "%~dp0\4_mmd.bat"

echo [1/7] Running MCD gland-prior experiments...
call "%~dp0\2_mcd.bat"

echo [4/7] Running HYBRID gland-prior experiments...
call "%~dp0\5_hybrid.bat"

echo [5/7] Running MCC gland-prior experiments...
call "%~dp0\6_mcc.bat"

echo [6/7] Running BNM gland-prior experiments...
call "%~dp0\7_bnm.bat"

echo [7/7] Running DAARDA gland-prior experiments...
call "%~dp0\8_daarda.bat"

echo.
echo ========================================================
echo ALL GLAND-PRIOR EXPERIMENTS COMPLETED!
echo ========================================================
pause
