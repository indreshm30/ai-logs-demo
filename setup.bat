@echo off
echo ========================================
echo AI-Driven Log Monitoring System Setup
echo ========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if Docker is running
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed or not running
    echo Please install Docker Desktop from https://docker.com
    pause
    exit /b 1
)

REM Create virtual environment
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment and install dependencies
echo Installing Python dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt

REM Start Docker services
echo Starting Docker services...
docker-compose up -d

REM Wait for services to start
echo Waiting for services to start (please wait 60 seconds)...
timeout /t 60 /nobreak >nul

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Open Terminal 1 and run: python consumer.py
echo 2. Open Terminal 2 and run: python producer.py
echo 3. Open browser to: http://localhost:3000
echo    Username: admin, Password: admin123
echo.
echo Press any key to exit...
pause >nul
