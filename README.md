# AI-Driven Log Monitoring System

A complete end-to-end demonstration of AI-driven log monitoring using Kafka, VictoriaMetrics, and Grafana.

## 🏗️ System Architecture

```
Log Producer → Kafka → Log Consumer (AI Classifier) → VictoriaMetrics → Grafana
     ↓                    ↓                              ↓              ↓
 Fake Logs          Rule-based AI              Metrics Storage    Visualization
                   Classification                                  & Alerting
```

## 📋 Components

1. **Log Producer** (`producer.py`): Generates realistic fake logs (INFO, WARN, ERROR, CRITICAL)
2. **Log Consumer** (`consumer.py`): Consumes logs, classifies them using AI rules, pushes metrics
3. **Kafka**: Message queue for log streaming
4. **VictoriaMetrics**: Time-series database for metrics storage
5. **Grafana**: Visualization and alerting dashboard

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed and running
- Python 3.8+ with pip
- Windows PowerShell (or any terminal)

### 1. Setup Python Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Infrastructure Services

```powershell
# Start all services (Kafka, VictoriaMetrics, Grafana)
docker-compose up -d

# Check if services are running
docker-compose ps
```

Wait for services to be healthy (usually 30-60 seconds).

### 3. Start Log Processing

Open **two separate terminal windows**:

**Terminal 1 - Start Consumer (AI Classifier):**
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start the log consumer
python consumer.py
```

**Terminal 2 - Start Producer (Log Generator):**
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start generating logs
python producer.py
```

### 4. Access Dashboards

- **Grafana Dashboard**: http://localhost:3000
  - Username: `admin`
  - Password: `admin123`
  - Dashboard: "AI-Driven Log Monitoring"

- **VictoriaMetrics**: http://localhost:8428

## 📊 Demo Scenarios

### Normal Operations
```powershell
python producer.py --interval 2 --batch-size 1
```

### Simulate Incident
```powershell
python producer.py --simulate-incident --incident-duration 60
```

### High Volume Testing
```powershell
python producer.py --interval 0.5 --batch-size 5
```

## 🤖 AI Classification Logic

The system uses a rule-based classifier that categorizes logs into:

- **Incident**: ERROR/CRITICAL logs or INFO logs with incident keywords
- **Warning**: WARN logs or INFO logs with warning keywords
- **Normal**: Regular INFO logs

### Keywords:
- **Incident**: `down`, `crash`, `failed`, `error`, `timeout`, `unavailable`, `memory`, `disk full`, `breach`, `overload`
- **Warning**: `high`, `slow`, `low`, `degrading`, `approaching`, `growing`, `nearly`, `expires`, `retry`

## 📈 Metrics Collected

- `log_incident_total`: Total number of incidents detected
- `log_warning_total`: Total number of warnings detected
- `log_total`: Total logs processed by level and service
- `log_processing_time_seconds`: Processing time per log classification

## 🚨 Grafana Alerts

- **High Incident Rate**: Triggers when incidents are detected (immediate)
- **High Warning Rate**: Triggers when >5 warnings occur in 5 minutes

## 🛠️ Troubleshooting

### Services won't start
```powershell
# Check Docker status
docker --version
docker-compose --version

# Restart services
docker-compose down
docker-compose up -d
```

### Python import errors
```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

### Kafka connection issues
```powershell
# Check Kafka logs
docker-compose logs kafka

# Wait for Kafka to fully start (may take 60+ seconds)
docker-compose logs kafka | findstr "started"
```

### No data in Grafana
1. Ensure both producer and consumer are running
2. Check VictoriaMetrics has data: http://localhost:8428/api/v1/label/__name__/values
3. Verify Grafana data source connection in Settings → Data Sources

## 📁 File Structure

```
aidemo/
├── docker-compose.yml          # Infrastructure setup
├── requirements.txt            # Python dependencies
├── producer.py                 # Log generator
├── consumer.py                 # AI classifier + metrics pusher
├── README.md                   # This file
└── grafana/
    ├── dashboards/
    │   └── log-monitoring.json # Grafana dashboard
    └── provisioning/
        ├── datasources/
        │   └── datasources.yml # VictoriaMetrics connection
        ├── dashboards/
        │   └── dashboard.yml   # Dashboard config
        └── alerting/
            └── rules.yml       # Alert rules
```

## 🔧 Configuration Options

### Producer Options
- `--kafka-servers`: Kafka bootstrap servers (default: localhost:9092)
- `--topic`: Kafka topic name (default: logs)
- `--interval`: Seconds between log batches (default: 2.0)
- `--batch-size`: Logs per batch (default: 1)
- `--simulate-incident`: Enable incident simulation mode
- `--incident-duration`: Incident simulation duration in seconds (default: 30)

### Consumer Options
- `--kafka-servers`: Kafka bootstrap servers (default: localhost:9092)
- `--kafka-topic`: Kafka topic to consume from (default: logs)
- `--victoria-metrics-url`: VictoriaMetrics URL (default: http://localhost:8428)
- `--consumer-group`: Kafka consumer group (default: log-monitor)
- `--stats-interval`: Statistics printing interval in seconds (default: 30)

## 🧪 Testing the System

1. **Start all services** as described in Quick Start
2. **Watch Grafana dashboard** for real-time metrics
3. **Trigger incidents** using `--simulate-incident` flag
4. **Observe alerts** in Grafana (Alerting section)
5. **Check metrics directly** at http://localhost:8428

## 🔄 Cleanup

```powershell
# Stop and remove all containers
docker-compose down -v

# Deactivate Python virtual environment
deactivate
```

## 📝 Notes

- The system is designed for demonstration purposes
- In production, replace rule-based classification with ML models
- Consider using proper message serialization (Avro/Protobuf)
- Add proper error handling and monitoring for production use
- Scale consumers horizontally for high-throughput scenarios

## 🆘 Support

If you encounter issues:
1. Check Docker Desktop is running
2. Verify ports 3000, 8428, 9092, 2181 are not in use
3. Ensure Python virtual environment is activated
4. Check service logs: `docker-compose logs [service-name]`
