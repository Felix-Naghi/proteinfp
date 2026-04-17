@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  ProteinFP — Windows setup script
REM  Run once from inside C:\Users\adria\Documents\proteinFP\
REM  Opens in VS Code automatically when done.
REM ─────────────────────────────────────────────────────────────────────────────

echo.
echo  ProteinFP Setup
echo  ═══════════════════════════════════════════════════
echo.

REM ── Check Python ─────────────────────────────────────────────────────────────
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo  [ERROR] Python not found. Install Python 3.11+ from python.org
    pause
    exit /b 1
)

FOR /F "tokens=2" %%i IN ('python --version 2^>^&1') DO SET PY_VER=%%i
echo  Python found: %PY_VER%

REM ── Create virtual environment ────────────────────────────────────────────────
IF NOT EXIST ".venv" (
    echo  Creating virtual environment...
    python -m venv .venv
    echo  Done.
) ELSE (
    echo  Virtual environment already exists — skipping creation.
)

REM ── Activate venv ─────────────────────────────────────────────────────────────
echo  Activating virtual environment...
call .venv\Scripts\activate.bat

REM ── Upgrade pip ──────────────────────────────────────────────────────────────
echo  Upgrading pip...
python -m pip install --upgrade pip setuptools wheel --quiet

REM ── Install PyTorch with CUDA 12.1 (RTX 5060 needs CUDA 12.x) ───────────────
echo.
echo  Installing PyTorch with CUDA 12.1 support for RTX 5060...
echo  (This may take a few minutes — PyTorch is ~2.5GB)
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM ── Install remaining requirements ───────────────────────────────────────────
echo.
echo  Installing pipeline dependencies...
pip install -r requirements.txt --quiet

REM ── Verify CUDA is detected ───────────────────────────────────────────────────
echo.
echo  Checking GPU / CUDA...
python -c "import torch; print('  CUDA available:', torch.cuda.is_available()); print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

REM ── Create data directories ───────────────────────────────────────────────────
echo.
echo  Creating data directories...
if not exist "data\input"        mkdir data\input
if not exist "data\structures"   mkdir data\structures
if not exist "data\intermediate" mkdir data\intermediate
if not exist "data\reports"      mkdir data\reports
if not exist "logs"              mkdir logs
if not exist "tests"             mkdir tests
echo  Done.

REM ── Run unit tests (offline, no network needed) ───────────────────────────────
echo.
echo  Running offline unit tests...
python -m pytest tests\test_01_fetch_structure.py -v -k "not Integration" --tb=short

echo.
echo  ═══════════════════════════════════════════════════
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
echo  ═══════════════════════════════════════════════════
echo.
pause
