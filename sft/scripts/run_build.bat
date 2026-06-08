@echo off
cd /d "%~dp0..\.."
echo [1/3] Building SFT datasets ...
python sft/build_dataset.py --config sft/conf/sft.conf %*
if errorlevel 1 (
    echo ERROR: build_dataset.py failed.
    pause
    exit /b 1
)
echo Done.
