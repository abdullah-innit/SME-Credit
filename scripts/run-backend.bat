@echo off
REM ── SME Credit — Run Backend ──
echo Starting backend on http://localhost:8000 ...
cd backend
call venv\Scripts\activate.bat
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
