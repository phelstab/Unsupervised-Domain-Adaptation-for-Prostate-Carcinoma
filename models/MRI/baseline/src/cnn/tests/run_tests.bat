@echo off
REM Run CNN data pipeline tests with uv venv
REM Usage: run_tests.bat

echo Running CNN data pipeline tests...
echo.

cd C:\workspace\uulm\PCa-classification

python -m uv run --python .venv-cnn python -m pytest models\MRI\baseline\src\cnn\tests\test_data_pipeline.py -v --tb=short

echo.
echo Tests completed!
pause
