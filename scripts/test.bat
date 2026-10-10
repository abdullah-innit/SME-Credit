@echo off
REM ── SME Credit — Test & Lint ──
echo ========================================
echo   SME Credit — Tests ^& Lint
echo ========================================

echo.
echo [1/2] Backend lint (ruff) ...
echo ----------------------------------------
cd backend
call venv\Scripts\activate.bat
ruff check app tests --fix
echo.
echo [2/2] Backend tests (pytest) ...
echo ----------------------------------------
pytest tests/ -v
cd ..

echo.
echo ========================================
echo   All checks complete!
echo ========================================
