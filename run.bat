@echo off
setlocal

rem -----------------------------------------------------------------------------
rem GLM-OCR local server launcher (Windows, venv)
rem - Creates/uses .venv next to this script
rem - Installs runtime deps (including transformers dev build)
rem - Starts FastAPI on configured host/port
rem -----------------------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "VENV_DIR=%SCRIPT_DIR%\.venv"
set "MODEL_CACHE_DIR=%SCRIPT_DIR%\models\hf_cache"
set "ENV_FILE=%SCRIPT_DIR%\.env"

if exist "%ENV_FILE%" (
    echo [+] Loading .env from "%ENV_FILE%"
    for /f "usebackq eol=# tokens=1* delims==" %%A in ("%ENV_FILE%") do (
        if not "%%A"=="" (
            set "%%A=%%B"
        )
    )
)

if not exist "%MODEL_CACHE_DIR%" (
    mkdir "%MODEL_CACHE_DIR%"
)

set "HF_HOME=%SCRIPT_DIR%\models\hf_home"
set "HF_HUB_CACHE=%MODEL_CACHE_DIR%"
set "TRANSFORMERS_CACHE=%MODEL_CACHE_DIR%"
set "GLM_MODEL_CACHE=%MODEL_CACHE_DIR%"
if "%TORCH_CHANNEL%"=="" set "TORCH_CHANNEL=cu126"

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [+] Creating virtual environment at "%VENV_DIR%" ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [!] Failed to create virtual environment. Ensure Python 3.10+ is installed and on PATH.
        exit /b 1
    )
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [!] Failed to activate virtual environment.
    exit /b 1
)

echo [+] Installing/ensuring dependencies...
python -m pip install --upgrade pip

echo [+] Installing PyTorch (%TORCH_CHANNEL%)...
if /I "%TORCH_CHANNEL%"=="cpu" (
    python -m pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cpu torch torchvision
) else (
    python -m pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/%TORCH_CHANNEL% torch torchvision
)
python -c "import torch; print('[torch]', torch.__version__, 'cuda=', torch.version.cuda, 'available=', torch.cuda.is_available())"

echo [+] Installing FastAPI and image/PDF dependencies...
python -m pip install fastapi uvicorn python-multipart pillow pypdfium2 accelerate

echo [+] Installing transformers (development build)...
python -m pip install git+https://github.com/huggingface/transformers.git

rem Configure host/port. Override by setting HOST/PORT before running.
if "%HOST%"=="" set "HOST=0.0.0.0"
if "%PORT%"=="" set "PORT=8000"

echo [+] Starting server at http://%HOST%:%PORT%
uvicorn app.main:app --host "%HOST%" --port "%PORT%"

endlocal
