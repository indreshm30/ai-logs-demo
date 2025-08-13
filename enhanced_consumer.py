# 🚀 Enhanced Consumer with AWS Integration and ML Model

import json
import logging
import os
import pickle
import requests
import time
from datetime import datetime, timedelta
from kafka import KafkaConsumer
from kafka.errors import KafkaError

# Import our custom modules
try:
    from aws_integration import AWSLogMonitor
    from email_alerts import IntegratedAlertSystem
    from ml_model_training import AdvancedLogClassifier
except ImportError:
    print("⚠️ Some modules not available, using fallback implementations")
    AWSLogMonitor = None
    IntegratedAlertSystem = None
    AdvancedLogClassifier = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedLogClassifier:
    """Enhanced log classifier with ML model support"""
    
    def __init__(self, use_ml_model=True):
        self.use_ml_model = use_ml_model
        self.ml_classifier = None
        
        # Load ML model if available
        if use_ml_model and os.path.exists("models/best_log_classifier.pkl"):
            try:
                self.ml_classifier = AdvancedLogClassifier()
                if self.ml_classifier.load_model():
                    logger.info("✅ ML model loaded successfully")
                else:
                    logger.warning("⚠️ Failed to load ML model, using rule-based fallback")
                    self.ml_classifier = None
            except Exception as e:
                logger.warning(f"⚠️ ML model loading error: {e}, using rule-based fallback")
                self.ml_classifier = None
        
        # Fallback rule-based classification
        self.incident_keywords = [
            'down', 'crash', 'failed', 'error', 'timeout', 'unavailable',
            'memory', 'disk full', 'breach', 'overload', 'unreachable',
            'critical', 'fatal', 'exception', 'deadlock', 'corrupted'
        ]
        
        self.warning_keywords = [
            'high', 'slow', 'retry', 'deprecated', 'load', 'warn',
            'approaching', 'exceeded', 'unusual', 'suspect', 'elevated'
        ]
    
    def classify_log(self, log_entry):
        """Classify log using ML model or rule-based approach"""
        try:
            # Extract message
            if isinstance(log_entry, dict):
                message = log_entry.get('message', '')
                level = log_entry.get('level', '').upper()
            else:
                message = str(log_entry)
                level = ''
            
            # Try ML model first
            if self.ml_classifier:
                try:
                    prediction = self.ml_classifier.predict_log(message)
                    classification = prediction['classification']
                    confidence = prediction['confidence']
                    
                    # Add confidence threshold
                    if confidence < 0.6:  # Low confidence, use rule-based fallback
                        classification = self._rule_based_classify(level, message)
                        method = 'rule-based (low ML confidence)'
                    else:
                        method = f'ml-model (confidence: {confidence:.3f})'
                    
                    logger.debug(f"Classification: {classification} via {method}")
                    return classification
                
                except Exception as e:
                    logger.debug(f"ML classification failed: {e}, using rule-based")
            
            # Fallback to rule-based
            return self._rule_based_classify(level, message)
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return 'normal'
    
    def _rule_based_classify(self, level, message):
        """Rule-based classification (fallback)"""
        message_lower = message.lower()
        
        # Check log level first
        if level in ['ERROR', 'CRITICAL', 'FATAL']:
            return 'incident'
        elif level in ['WARN', 'WARNING']:
            return 'warning'
        elif level == 'INFO':
            # Check for incident indicators in INFO messages
            if any(keyword in message_lower for keyword in self.incident_keywords):
                return 'incident'
            elif any(keyword in message_lower for keyword in self.warning_keywords):
                return 'warning'
        
        return 'normal'

class EnhancedMetricsPusher:
    """Enhanced metrics pusher with additional AWS metrics"""
    
    def __init__(self, victoria_metrics_url='http://localhost:8428'):
        self.base_url = victoria_metrics_url
        self.session = requests.Session()
        
        # Initialize counters
        self.incident_count = 0
        self.warning_count = 0
        self.total_logs_count = 0
        self.aws_logs_count = 0
        
        # Track metrics by source
        self.metrics_by_source = {}
    
    def push_metric(self, metric_name, value, labels=None):
        """Push a single metric to VictoriaMetrics"""
        try:
            labels = labels or {}
            labels_str = ','.join([f'{k}="{v}"' for k, v in labels.items()])
            
            if labels_str:
                metric_line = f'{metric_name}{{{labels_str}}} {value}'
            else:
                metric_line = f'{metric_name} {value}'
            
            url = f'{self.base_url}/api/v1/import/prometheus'
            response = self.session.post(url, data=metric_line)
            
            if response.status_code == 204:
                logger.debug(f"Metric pushed: {metric_name} = {value}")
                return True
            else:
                logger.error(f"Failed to push metric: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error pushing metric: {e}")
            return False
    
    def update_counters(self, classification, source='local', service='unknown'):
        """Update counters and push metrics"""
        timestamp = int(time.time())
        
        # Update global counters
        self.total_logs_count += 1
        
        if classification == 'incident':
            self.incident_count += 1
            self.push_metric('log_incident_total', self.incident_count)
            self.push_metric('log_incident_rate', 1, {'source': source, 'service': service})
            
        elif classification == 'warning':
            self.warning_count += 1 
            self.push_metric('log_warning_total', self.warning_count)
            self.push_metric('log_warning_rate', 1, {'source': source, 'service': service})
        
        # Track by source
        if source == 'aws':
            self.aws_logs_count += 1
            self.push_metric('log_aws_total', self.aws_logs_count)
        
        # Update source-specific metrics
        source_key = f"{source}_{service}"
        if source_key not in self.metrics_by_source:
            self.metrics_by_source[source_key] = {'incidents': 0, 'warnings': 0, 'total': 0}
        
        self.metrics_by_source[source_key]['total'] += 1
        if classification == 'incident':
            self.metrics_by_source[source_key]['incidents'] += 1
        elif classification == 'warning':
            self.metrics_by_source[source_key]['warnings'] += 1
        
        # Push source-specific metrics
        self.push_metric('log_total_by_source', self.metrics_by_source[source_key]['total'], 
                        {'source': source, 'service': service})
        
        # Push overall totals
        self.push_metric('log_total', self.total_logs_count)

class EnhancedLogConsumer:
    """Enhanced log consumer with AWS integration and ML classification"""
    
    def __init__(self):
        self.classifier = EnhancedLogClassifier(use_ml_model=True)
        self.metrics_pusher = EnhancedMetricsPusher()
        
        # Initialize AWS integration
        self.aws_monitor = None
        if AWSLogMonitor:
            try:
                self.aws_monitor = AWSLogMonitor()
                logger.info("✅ AWS integration initialized")
            except Exception as e:
                logger.warning(f"⚠️ AWS integration failed: {e}")
        
        # Initialize alert system
        self.alert_system = None
        if IntegratedAlertSystem:
            try:
                self.alert_system = IntegratedAlertSystem()
                logger.info("✅ Alert system initialized")
            except Exception as e:
                logger.warning(f"⚠️ Alert system failed: {e}")
        
        # Alert thresholds
        self.incident_threshold = 1  # Send alert on any incident
        self.warning_threshold = 5   # Send alert after 5 warnings
        self.last_alert_time = {'incident': 0, 'warning': 0}
        self.alert_cooldown = 300    # 5 minutes cooldown
        
        # Performance tracking
        self.processing_times = []
        self.last_aws_fetch = 0
        self.aws_fetch_interval = 300  # Fetch AWS logs every 5 minutes
    
    def setup_kafka_consumer(self):
        """Setup Kafka consumer"""
        try:
            consumer = KafkaConsumer(
                'logs',
                bootstrap_servers=['localhost:9092'],
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest',
                group_id='log-consumer-enhanced'
            )
            logger.info("✅ Kafka consumer setup successful")
            return consumer
        except Exception as e:
            logger.error(f"❌ Failed to setup Kafka consumer: {e}")
            return None
    
    def fetch_aws_logs(self):
        """Fetch logs from AWS and process them"""
        if not self.aws_monitor:
            return
        
        current_time = time.time()
        if current_time - self.last_aws_fetch < self.aws_fetch_interval:
            return
        
        try:
            logger.info("🔄 Fetching AWS logs...")
            aws_logs = self.aws_monitor.get_all_aws_logs()
            
            for log_entry in aws_logs:
                start_time = time.time()
                
                # Classify AWS log
                classification = self.classifier.classify_log(log_entry)
                
                # Extract service/source info
                source = 'aws'
                service = log_entry.get('log_group', 'unknown').replace('/aws/', '').replace('/', '_')
                
                # Update metrics
                self.metrics_pusher.update_counters(classification, source, service)
                
                # Display result
                emoji = self.get_classification_emoji(classification)
                print(f"{emoji} AWS {classification.upper()}: {log_entry.get('message', '')[:100]}... (Source: {service})")
                
                # Check for alerts
                self.check_and_send_alerts(classification, log_entry)
                
                # Track performance
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
            
            self.last_aws_fetch = current_time
            logger.info(f"✅ Processed {len(aws_logs)} AWS logs")
            
        except Exception as e:
            logger.error(f"❌ Error fetching AWS logs: {e}")
    
    def get_classification_emoji(self, classification):
        """Get emoji for classification"""
        emojis = {
            'incident': '🚨',
            'warning': '⚠️', 
            'normal': '✅'
        }
        return emojis.get(classification, '📝')
    
    def check_and_send_alerts(self, classification, log_entry):
        """Check if alerts should be sent"""
        if not self.alert_system:
            return
        
        current_time = time.time()
        
        if classification == 'incident':
            # Send incident alert immediately (with cooldown)
            if current_time - self.last_alert_time['incident'] > self.alert_cooldown:
                incident_data = {
                    'timestamp': datetime.now().isoformat(),
                    'severity': 'CRITICAL',
                    'classification': 'System Incident',
                    'message': log_entry.get('message', 'Unknown incident'),
                    'source': log_entry.get('source', 'Unknown'),
                    'total_incidents': self.metrics_pusher.incident_count,
                    'recent_warnings': self.metrics_pusher.warning_count,
                    'system_health': 'DEGRADED'
                }
                
                if self.alert_system.email_alerts.send_incident_alert(incident_data):
                    self.last_alert_time['incident'] = current_time
                    logger.info("📧 Incident alert sent via email")
        
        elif classification == 'warning':
            # Send warning alert after threshold
            if (self.metrics_pusher.warning_count % self.warning_threshold == 0 and
                current_time - self.last_alert_time['warning'] > self.alert_cooldown):
                
                warning_data = {
                    'timestamp': datetime.now().isoformat(),
                    'warning_type': 'High Warning Rate',
                    'message': f"{self.warning_threshold} warnings detected - preventive action recommended",
                    'source': log_entry.get('source', 'Unknown'),
                    'count_5min': self.warning_threshold,
                    'total_warnings': self.metrics_pusher.warning_count,
                    'risk_level': 'Medium',
                    'trend': 'Increasing'
                }
                
                if self.alert_system.email_alerts.send_warning_alert(warning_data):
                    self.last_alert_time['warning'] = current_time
                    logger.info("📧 Warning alert sent via email")
    
    def display_stats(self):
        """Display current statistics"""
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Calculate average processing time
        avg_processing_time = 0
        if self.processing_times:
            recent_times = self.processing_times[-100:]  # Last 100 logs
            avg_processing_time = sum(recent_times) / len(recent_times) * 1000  # in ms
        
        stats_message = (
            f"📊 [{current_time}] ENHANCED STATS - "
            f"🚨 Incidents: {self.metrics_pusher.incident_count}, "
            f"⚠️ Warnings: {self.metrics_pusher.warning_count}, "
            f"📝 Total: {self.metrics_pusher.total_logs_count}, "
            f"☁️ AWS Logs: {self.metrics_pusher.aws_logs_count}, "
            f"⏱️ Avg Processing: {avg_processing_time:.1f}ms"
        )
        
        print(stats_message)
    
    def run(self):
        """Main consumer loop"""
        logger.info("🚀 Starting Enhanced Log Consumer...")
        
        # Setup Kafka consumer
        consumer = self.setup_kafka_consumer()
        if not consumer:
            logger.error("❌ Failed to setup consumer, exiting")
            return
        
        logger.info("✅ Enhanced consumer ready - processing logs from Kafka + AWS")
        
        try:
            stats_counter = 0
            
            for message in consumer:
                try:
                    start_time = time.time()
                    
                    # Process Kafka message
                    log_entry = message.value
                    classification = self.classifier.classify_log(log_entry)
                    
                    # Update metrics
                    source = log_entry.get('source', 'kafka')
                    service = log_entry.get('service', 'unknown')
                    self.metrics_pusher.update_counters(classification, source, service)
                    
                    # Display result
                    emoji = self.get_classification_emoji(classification)
                    print(f"{emoji} {classification.upper()}: {log_entry.get('message', 'No message')}")
                    
                    # Check for alerts
                    self.check_and_send_alerts(classification, log_entry)
                    
                    # Track performance
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)
                    
                    # Fetch AWS logs periodically
                    self.fetch_aws_logs()
                    
                    # Display stats every 10 messages
                    stats_counter += 1
                    if stats_counter >= 10:
                        self.display_stats()
                        stats_counter = 0
                    
                except Exception as e:
                    logger.error(f"❌ Error processing message: {e}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("⏹️ Consumer stopped by user")
        except Exception as e:
            logger.error(f"❌ Consumer error: {e}")
        finally:
            consumer.close()
            logger.info("👋 Enhanced consumer shutdown complete")

def main():
    """Main function"""
    print("🚀 Enhanced AI Log Consumer")
    print("=" * 50)
    print("Features:")
    print("✅ ML-powered log classification")  
    print("☁️ AWS CloudWatch integration")
    print("📧 Email & Slack alerts")
    print("📊 Advanced metrics tracking")
    print("⚡ Real-time processing")
    print("=" * 50)
    
    # Check dependencies
    missing_deps = []
    try:
        import pandas
        import sklearn
        import boto3
    except ImportError as e:
        missing_deps.append(str(e))
    
    if missing_deps:
        print("⚠️ Missing dependencies - some features may be limited:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("💡 Install with: pip install pandas scikit-learn boto3")
        print()
    
    # Start consumer
    consumer = EnhancedLogConsumer()
    consumer.run()

if __name__ == "__main__":
    main()
