@echo off
REM ── SME Credit — Full Setup ──
echo ========================================
echo   SME Credit — Setup
echo ========================================

REM Copy .env if missing
if not exist ".env" (
    echo Creating .env from .env.example ...
    copy .env.example .env
)

echo.
echo [1/2] Setting up Backend ...
echo ----------------------------------------
cd backend
if not exist "venv" (
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install -r requirements.txt --quiet
cd ..

echo.
echo [2/2] Setting up Frontend ...
echo ----------------------------------------
cd frontend
call npm install
cd ..

echo.
echo ========================================
echo   Setup complete!
echo   Run: .\scripts\run-all.bat
echo ========================================
