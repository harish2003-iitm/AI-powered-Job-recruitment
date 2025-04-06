@echo off
echo =====================================
echo   AI Recruitment System Setup
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

:: Install backend dependencies
echo Installing backend dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Failed to install backend dependencies.
    exit /b 1
)
echo Backend dependencies installed successfully.

:: Install frontend dependencies
echo.
echo Installing frontend dependencies...
cd frontend && npm install
if %ERRORLEVEL% neq 0 (
    echo Failed to install frontend dependencies.
    exit /b 1
)
echo Frontend dependencies installed successfully.

echo.
echo =====================================
echo Setup completed successfully!
echo.
echo To start the application, run:
echo   start.bat
echo =====================================

pause 