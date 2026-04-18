@echo off
chcp 65001 >nul
echo.
echo  ProteinFP Setup
echo  ===================================================
echo.

REM -- Check Python is installed ------------------------------------
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo  [ERROR] Python not found.
    echo  Install Python 3.12 from https://www.python.org/downloads/release/python-3120/
    echo  Make sure to tick "Add Python to PATH" during install.
    pause
    exit /b 1
)

REM -- Check Python version is 3.12 (PyTorch max supported) ---------
FOR /F "tokens=2" %%i IN ('python --version 2^>^&1') DO SET PY_VER=%%i
echo  Python found: %PY_VER%

python -c "import sys; v=sys.version_info; ok=(v.major==3 and v.minor==12); print('OK' if ok else 'WRONG')" > %TEMP%\pycheck.txt 2>&1
SET /P PY_OK=<%TEMP%\pycheck.txt

IF NOT "%PY_OK%"=="OK" (
    echo.
    echo  [ERROR] PyTorch requires Python 3.12 but you have Python %PY_VER%
    echo.
    echo  Python 3.14 is too new -- PyTorch has no wheels for it yet.
    echo.
    echo  Fix options:
    echo    OPTION A ^(recommended^) -- Install Python 3.12 alongside your current Python:
    echo      1. Go to https://www.python.org/downloads/release/python-3120/
    echo      2. Download "Windows installer ^(64-bit^)"
    echo      3. Install it -- do NOT uninstall 3.14, just add 3.12 alongside it
    echo      4. Re-run this script using:  py -3.12 -m venv .venv
    echo         then manually run:         .venv\Scripts\activate
    echo         then skip to the pip steps below
    echo.
    echo    OPTION B -- Use the py launcher to create the venv with 3.12:
    echo      py -3.12 -m venv .venv
    echo.
    echo  After installing Python 3.12, re-run this script.
    pause
    exit /b 1
)

REM -- Create virtual environment ------------------------------------
IF NOT EXIST ".venv" (
    echo  Creating virtual environment with Python 3.12...
    python -m venv .venv
    echo  Done.
) ELSE (
    echo  Virtual environment already exists -- skipping.
)

REM -- Activate venv -------------------------------------------------
echo  Activating virtual environment...
call .venv\Scripts\activate.bat

REM -- Upgrade pip ---------------------------------------------------
echo  Upgrading pip...
python -m pip install --upgrade pip setuptools wheel --quiet

REM -- Install PyTorch with CUDA 12.8 (correct for RTX 5060) --------
echo.
echo  Installing PyTorch with CUDA 12.8 for RTX 5060...
echo  (This may take several minutes -- PyTorch is ~2.5GB)
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

REM -- Verify PyTorch installed correctly ---------------------------
echo.
echo  Verifying PyTorch + CUDA...
python -c "import torch; print('  PyTorch version:', torch.__version__); print('  CUDA available:', torch.cuda.is_available()); print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected -- check drivers')"

REM -- Install remaining requirements --------------------------------
echo.
echo  Installing pipeline dependencies...
pip install -r requirements.txt --quiet

REM -- Create data directories ---------------------------------------
echo.
echo  Creating data directories...
if not exist "data\input"        mkdir data\input
if not exist "data\structures"   mkdir data\structures
if not exist "data\intermediate" mkdir data\intermediate
if not exist "data\reports"      mkdir data\reports
if not exist "logs"              mkdir logs
if not exist "tests"             mkdir tests
echo  Done.

REM -- Run offline unit tests ----------------------------------------
echo.
echo  Running offline unit tests...
python -m pytest tests\test_01_fetch_structure.py -v -k "not Integration" --tb=short

echo.
echo  ===================================================
echo  Setup complete.
echo.
echo  To activate the environment in future terminals:
echo    .venv\Scripts\activate
echo.
echo  To run Module 01 on a protein:
echo    python pipeline\01_fetch_structure.py --uniprot P04637
echo.
echo  To open in VS Code:
echo    code .
echo  ===================================================
echo.
pause