@echo off
:: Main runner: Execute all UDA algorithm experiments for public dataset splits 2
cd /d "%~dp0..\..\.."
echo ========================================================
echo MAIN UDA EXPERIMENT RUNNER (Public Splits 2)
echo Working directory: %CD%
echo Split: RUMC_to_PCNN, RUMC_to_ZGT (2-split mode)
echo Algorithms: MCD, DANN, MMD, HYBRID, MCC, BNM, DAARDA
echo Total full-reg runs: 7 algorithms x 3 DA weights x 2 pairs = 42 experiments
echo ========================================================
echo.

echo [1/7] Running MCD experiments...
call "%~dp0\2_mcd.bat"

echo [2/7] Running DANN experiments...
call "%~dp0\3_dann.bat"

echo [3/7] Running MMD experiments...
call "%~dp0\4_mmd.bat"

echo [4/7] Running HYBRID experiments...
call "%~dp0\5_hybrid.bat"

echo [5/7] Running MCC experiments...
call "%~dp0\6_mcc.bat"

echo [6/7] Running BNM experiments...
call "%~dp0\7_bnm.bat"

echo [7/7] Running DAARDA experiments...
call "%~dp0\8_daarda.bat"

echo.
echo ========================================================
echo ALL PUBLIC SPLITS 2 EXPERIMENTS COMPLETED!
echo ========================================================
pause
