@echo off
echo =====================================
echo   AI-Powered Recruitment System
echo =====================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed.
    exit /b 1
)

:: Check if Node.js is installed
node --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Node.js is not installed.
    exit /b 1
)

:: Start the backend server in a new window
echo Starting backend server...
start "AI Recruitment Backend" cmd /c "python main.py"

:: Wait for backend to start up
echo Waiting for backend to start up...
timeout /t 3 /nobreak >nul

:: Start the frontend development server in a new window
echo Starting frontend server...
start "AI Recruitment Frontend" cmd /c "cd frontend && npm start"

echo.
echo =====================================
echo Servers are now running:
echo   Backend: http://localhost:8000
echo   Frontend: http://localhost:3000
echo.
echo Close the server windows to stop the servers.
echo =====================================

:: Keep the command window open
pause 