@echo off
cd /d "%~dp0..\.."
echo [3/3] Predicting and evaluating ...
python sft/predict.py --config sft/conf/sft.conf %*
if errorlevel 1 (
    echo ERROR: predict.py failed.
    pause
    exit /b 1
)
echo Done.
