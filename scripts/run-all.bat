@echo off
REM ── SME Credit — Run All ──
echo ========================================
echo   SME Credit — Starting All Services
echo ========================================

echo Starting backend in background ...
start "SME Credit Backend" cmd /k "cd backend && call venv\Scripts\activate.bat && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

echo Starting frontend ...
start "SME Credit Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo   Backend:  http://localhost:8000/docs
echo   Frontend: http://localhost:3000
echo   Health:   http://localhost:8000/api/v1/health
echo ========================================
