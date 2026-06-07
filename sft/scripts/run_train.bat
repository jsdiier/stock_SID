@echo off
cd /d "%~dp0.."
echo [2/3] Training SFT model ...
python train.py --config conf/sft.conf %*
if errorlevel 1 (
    echo ERROR: train.py failed.
    pause
    exit /b 1
)
echo Done.
