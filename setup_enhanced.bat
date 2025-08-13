@echo off
REM 🚀 Enhanced AI Log Monitoring Setup & Demo Script (Windows)

echo 🚀 Enhanced AI Log Monitoring System Setup
echo ==========================================

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
) else (
    echo ✅ Python found
)

REM Check Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker not found. Please install Docker Desktop first.
    pause
    exit /b 1
) else (
    echo ✅ Docker found
)

REM Check Docker Compose
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose not found. Please install Docker Compose first.
    pause
    exit /b 1
) else (
    echo ✅ Docker Compose found
)

echo.

REM Setup Python environment
echo 🐍 Setting up Python virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📚 Installing Python dependencies...
pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo ✅ All dependencies installed successfully
) else (
    echo ⚠️ Some dependencies may have failed to install
    echo 💡 Try running manually: pip install pandas scikit-learn boto3 matplotlib seaborn
)

echo.

REM Train ML model
echo 🤖 Training ML models...
if exist "ml_model_training.py" (
    echo 🎯 Starting model training ^(this may take a few minutes^)...
    python ml_model_training.py
    
    if %errorlevel% equ 0 (
        echo ✅ ML models trained successfully
        echo 📁 Models saved in .\models\ directory
    ) else (
        echo ⚠️ Model training had issues - will use rule-based classification
    )
) else (
    echo ⚠️ ML training script not found - will use rule-based classification
)

echo.

REM Setup AWS (optional)
echo ☁️ AWS Setup ^(Optional^)
echo =======================
echo To enable AWS log monitoring, you need to:
echo 1. Install AWS CLI: https://aws.amazon.com/cli/
echo 2. Configure credentials: aws configure
echo 3. Set up IAM permissions for CloudWatch Logs
echo.
echo For now, skipping AWS setup. You can enable it later.
echo.

REM Setup email alerts (optional)
echo 📧 Email Alerts Setup ^(Optional^)
echo ===============================
echo To enable email alerts, set these environment variables:
echo set EMAIL_USERNAME=your-email@gmail.com
echo set EMAIL_PASSWORD=your-app-password
echo set SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
echo.
echo For Gmail, use App Passwords: https://support.google.com/accounts/answer/185833
echo.

REM Start infrastructure
echo 🏗️ Starting Docker infrastructure...
docker-compose up -d

if %errorlevel% equ 0 (
    echo ✅ Docker services started successfully
    
    REM Wait for services to be ready
    echo ⏳ Waiting for services to be ready...
    timeout /t 30 /nobreak > nul
    
    REM Check service status
    echo 🔍 Checking service status...
    curl -s http://localhost:9092 > nul 2>&1 && echo   - Kafka: ✅ Ready || echo   - Kafka: ❌ Not ready
    curl -s http://localhost:8428 > nul 2>&1 && echo   - VictoriaMetrics: ✅ Ready || echo   - VictoriaMetrics: ❌ Not ready
    curl -s http://localhost:3000 > nul 2>&1 && echo   - Grafana: ✅ Ready || echo   - Grafana: ❌ Not ready
    
) else (
    echo ❌ Failed to start Docker services
    pause
    exit /b 1
)

echo.

REM Final instructions
echo 🎉 Setup Complete!
echo ==================
echo.
echo 🚀 Ready to run enhanced demo:
echo.
echo 1️⃣  Start Enhanced Consumer ^(ML + AWS + Alerts^):
echo    python enhanced_consumer.py
echo.
echo 2️⃣  Start Original Producer ^(in another terminal^):
echo    python producer.py
echo.
echo 3️⃣  Open Grafana Dashboard:
echo    http://localhost:3000 ^(admin/admin123^)
echo.
echo 🔧 Available Features:
echo   ✅ Original Kafka log streaming
echo   ✅ ML-powered log classification
echo   ✅ AWS CloudWatch integration ^(if configured^)  
echo   ✅ Email/Slack alerts ^(if configured^)
echo   ✅ Enhanced metrics tracking
echo   ✅ Real-time Grafana dashboards
echo.
echo 💡 Tips:
echo   - Train custom models with your own log data
echo   - Configure AWS credentials for cloud log monitoring
echo   - Set up email alerts for production use
echo   - Monitor performance metrics in Grafana
echo.
echo 📚 Next Steps:
echo   - Check PRODUCTION_ROADMAP.md for scaling plan
echo   - Review aws_integration.py for cloud setup
echo   - Customize email_alerts.py for your needs
echo.
echo 🎯 Demo is ready! Start the enhanced consumer and producer to see the magic! ✨

pause
