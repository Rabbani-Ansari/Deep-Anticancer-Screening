@echo off
echo ===================================================
echo   Starting OncoScreen AI - Modern Web Interface
echo ===================================================

echo [1/2] Starting Backend Server (FastAPI)...
start "OncoScreen Backend" cmd /k "cd backend && uvicorn main:app --reload"

echo [2/2] Starting Frontend Interface (React)...
start "OncoScreen Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo 🚀 Systems launching! 
echo.
echo Backend URL: http://localhost:8000/docs
echo Frontend URL: http://localhost:5173
echo.
echo Press any key to exit this launcher...
pause
