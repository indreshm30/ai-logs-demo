#!/usr/bin/env python3
"""
Setup script for AI-Driven Log Monitoring Demo
Helps verify environment and setup requirements
"""

import sys
import subprocess
import os
import time
import requests
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"❌ {description} - Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - Error: {e}")
        return False

def check_port(port, service_name):
    """Check if a port is accessible"""
    try:
        response = requests.get(f"http://localhost:{port}", timeout=5)
        print(f"✅ {service_name} (port {port}) - Accessible")
        return True
    except requests.exceptions.RequestException:
        print(f"❌ {service_name} (port {port}) - Not accessible")
        return False

def main():
    """Main setup and verification function"""
    print("🚀 AI-Driven Log Monitoring System Setup")
    print("=" * 50)
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    # Check Docker
    if not run_command("docker --version", "Checking Docker"):
        print("❌ Docker is required. Please install Docker Desktop")
        return False
    
    if not run_command("docker-compose --version", "Checking Docker Compose"):
        print("❌ Docker Compose is required")
        return False
    
    # Check if virtual environment exists
    venv_path = Path("venv")
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        if not run_command("python -m venv venv", "Creating virtual environment"):
            return False
    else:
        print("✅ Virtual environment exists")
    
    # Install Python dependencies
    print("📦 Installing Python dependencies...")
    if os.name == 'nt':  # Windows
        activate_cmd = r".\venv\Scripts\activate && pip install -r requirements.txt"
    else:  # Unix/Linux/Mac
        activate_cmd = "source venv/bin/activate && pip install -r requirements.txt"
    
    if not run_command(activate_cmd, "Installing Python packages"):
        print("❌ Failed to install Python packages")
        return False
    
    # Start Docker services
    print("🐳 Starting Docker services...")
    if not run_command("docker-compose up -d", "Starting Docker Compose services"):
        print("❌ Failed to start Docker services")
        return False
    
    # Wait for services to start
    print("⏳ Waiting for services to start (60 seconds)...")
    time.sleep(60)
    
    # Check service health
    print("\n🔍 Checking service health:")
    services = [
        (3000, "Grafana"),
        (8428, "VictoriaMetrics"),
        (9092, "Kafka")
    ]
    
    all_healthy = True
    for port, service in services:
        if not check_port(port, service):
            all_healthy = False
    
    if all_healthy:
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Activate virtual environment:")
        if os.name == 'nt':
            print("   .\\venv\\Scripts\\Activate.ps1")
        else:
            print("   source venv/bin/activate")
        print("2. Start consumer: python consumer.py")
        print("3. Start producer: python producer.py")
        print("4. Open Grafana: http://localhost:3000 (admin/admin123)")
        return True
    else:
        print("\n⚠️ Some services are not healthy. Please check Docker logs:")
        print("   docker-compose logs")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
