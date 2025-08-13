#!/bin/bash

# 🚀 Enhanced AI Log Monitoring Setup & Demo Script

echo "🚀 Enhanced AI Log Monitoring System Setup"
echo "=========================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Python
if command_exists python; then
    PYTHON_CMD=python
elif command_exists python3; then
    PYTHON_CMD=python3
else
    echo "❌ Python not found. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python found: $($PYTHON_CMD --version)"

# Check Docker
if command_exists docker; then
    echo "✅ Docker found: $(docker --version | head -n1)"
else
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if command_exists docker-compose; then
    echo "✅ Docker Compose found: $(docker-compose --version)"
else
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo ""

# Setup Python environment
echo "🐍 Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate 2>/dev/null

# Install/upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ All dependencies installed successfully"
else
    echo "⚠️ Some dependencies may have failed to install"
    echo "💡 Try running: pip install pandas scikit-learn boto3 matplotlib seaborn"
fi

echo ""

# Train ML model
echo "🤖 Training ML models..."
if [ -f "ml_model_training.py" ]; then
    echo "🎯 Starting model training (this may take a few minutes)..."
    $PYTHON_CMD ml_model_training.py
    
    if [ $? -eq 0 ]; then
        echo "✅ ML models trained successfully"
        echo "📁 Models saved in ./models/ directory"
    else
        echo "⚠️ Model training had issues - will use rule-based classification"
    fi
else
    echo "⚠️ ML training script not found - will use rule-based classification"
fi

echo ""

# Setup AWS (optional)
echo "☁️ AWS Setup (Optional)"
echo "======================="
echo "To enable AWS log monitoring, you need to:"
echo "1. Install AWS CLI: https://aws.amazon.com/cli/"
echo "2. Configure credentials: aws configure"
echo "3. Set up IAM permissions for CloudWatch Logs"
echo ""
echo "For now, skipping AWS setup. You can enable it later."
echo ""

# Setup email alerts (optional)
echo "📧 Email Alerts Setup (Optional)"
echo "==============================="
echo "To enable email alerts, set these environment variables:"
echo "export EMAIL_USERNAME='your-email@gmail.com'"
echo "export EMAIL_PASSWORD='your-app-password'"
echo "export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...'"
echo ""
echo "For Gmail, use App Passwords: https://support.google.com/accounts/answer/185833"
echo ""

# Start infrastructure
echo "🏗️ Starting Docker infrastructure..."
docker-compose up -d

if [ $? -eq 0 ]; then
    echo "✅ Docker services started successfully"
    
    # Wait for services to be ready
    echo "⏳ Waiting for services to be ready..."
    sleep 30
    
    # Check service status
    echo "🔍 Checking service status..."
    echo "  - Kafka: $(curl -s http://localhost:9092 && echo "✅ Ready" || echo "❌ Not ready")"
    echo "  - VictoriaMetrics: $(curl -s http://localhost:8428 && echo "✅ Ready" || echo "❌ Not ready")"
    echo "  - Grafana: $(curl -s http://localhost:3000 && echo "✅ Ready" || echo "❌ Not ready")"
    
else
    echo "❌ Failed to start Docker services"
    exit 1
fi

echo ""

# Final instructions
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "🚀 Ready to run enhanced demo:"
echo ""
echo "1️⃣  Start Enhanced Consumer (ML + AWS + Alerts):"
echo "   $PYTHON_CMD enhanced_consumer.py"
echo ""
echo "2️⃣  Start Original Producer (in another terminal):"
echo "   $PYTHON_CMD producer.py"
echo ""
echo "3️⃣  Open Grafana Dashboard:"
echo "   http://localhost:3000 (admin/admin123)"
echo ""
echo "🔧 Available Features:"
echo "  ✅ Original Kafka log streaming"
echo "  ✅ ML-powered log classification"
echo "  ✅ AWS CloudWatch integration (if configured)"  
echo "  ✅ Email/Slack alerts (if configured)"
echo "  ✅ Enhanced metrics tracking"
echo "  ✅ Real-time Grafana dashboards"
echo ""
echo "💡 Tips:"
echo "  - Train custom models with your own log data"
echo "  - Configure AWS credentials for cloud log monitoring"
echo "  - Set up email alerts for production use"
echo "  - Monitor performance metrics in Grafana"
echo ""
echo "📚 Next Steps:"
echo "  - Check PRODUCTION_ROADMAP.md for scaling plan"
echo "  - Review aws_integration.py for cloud setup"
echo "  - Customize email_alerts.py for your needs"
echo ""
echo "🎯 Demo is ready! Start the enhanced consumer and producer to see the magic! ✨"
