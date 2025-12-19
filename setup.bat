@echo off
REM Simple Windows Batch Setup Script for CodeLlama Fine-Tuning
REM Usage: Double-click setup.bat or run in Command Prompt

echo ========================================
echo CodeLlama Fine-Tuning Setup (Windows)
echo ========================================
echo.

REM Check if Python is installed
echo Step 1: Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo Please install Python 3.10 from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)
python --version
echo [OK] Python found
echo.

REM Check NVIDIA GPU
echo Step 2: Checking NVIDIA GPU...
nvidia-smi --query-gpu=name --format=csv,noheader >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] NVIDIA GPU or drivers not found!
    echo Please install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx
    pause
    exit /b 1
)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo [OK] NVIDIA GPU detected
echo.

REM Check CUDA
echo Step 3: Checking CUDA...
nvcc --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] CUDA toolkit not found
    echo Please install CUDA 12.1 from: https://developer.nvidia.com/cuda-downloads
    echo After installation, restart this script
    pause
    exit /b 1
)
echo [OK] CUDA found
nvcc --version | findstr "release"
echo.

REM Create virtual environment
echo Step 4: Creating virtual environment...
if exist venv (
    echo [INFO] Virtual environment already exists
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)
echo.

REM Activate virtual environment
echo Step 5: Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo Step 6: Upgrading pip...
python -m pip install --upgrade pip setuptools wheel --quiet
echo [OK] pip upgraded
echo.

REM Install PyTorch
echo Step 7: Installing PyTorch with CUDA support...
echo This may take several minutes...
echo.
echo Select CUDA version:
echo   [1] CUDA 12.1 (Recommended)
echo   [2] CUDA 11.8
echo.
set /p cuda_choice="Enter choice (1 or 2): "

if "%cuda_choice%"=="1" (
    echo Installing PyTorch for CUDA 12.1...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else if "%cuda_choice%"=="2" (
    echo Installing PyTorch for CUDA 11.8...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo [ERROR] Invalid choice
    pause
    exit /b 1
)

if %errorlevel% neq 0 (
    echo [ERROR] PyTorch installation failed
    pause
    exit /b 1
)
echo [OK] PyTorch installed
echo.

REM Verify PyTorch CUDA
echo Step 8: Verifying PyTorch CUDA support...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
if %errorlevel% neq 0 (
    echo [ERROR] PyTorch CUDA verification failed
    pause
    exit /b 1
)
echo [OK] PyTorch CUDA working
echo.

REM Check for requirements.txt
echo Step 9: Installing dependencies...
if not exist requirements.txt (
    echo [ERROR] requirements.txt not found!
    echo Please ensure requirements.txt is in the current directory
    pause
    exit /b 1
)

echo Installing packages from requirements.txt...
echo This will take 5-10 minutes...
pip install -r requirements.txt --quiet
echo [OK] Dependencies installed
echo.

REM Install Windows-specific bitsandbytes
echo Step 10: Installing bitsandbytes (Windows version)...
pip uninstall bitsandbytes -y >nul 2>&1
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl --quiet
echo [OK] bitsandbytes installed
echo.

REM Final verification
echo Step 11: Verifying all installations...
python -c "import torch, transformers, datasets, peft, trl, accelerate; print('[OK] All core libraries verified')"
if %errorlevel% neq 0 (
    echo [WARNING] Some libraries may not be installed correctly
)
echo.

REM Display system info
echo ========================================
echo System Information:
echo ========================================
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('VRAM:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB' if torch.cuda.is_available() else 'N/A')"
echo.

echo ========================================
echo [SUCCESS] Setup completed!
echo ========================================
echo.
echo Next steps:
echo   1. Generate dataset: python build_js_dataset.py
echo   2. Start fine-tuning: python finetune_codellama_js.py
echo.
echo Tips:
echo   - To activate venv: venv\Scripts\activate.bat
echo   - Monitor GPU: nvidia-smi
echo   - Monitor training: tensorboard --logdir results
echo.
echo Press any key to exit...
pause >nul
