#!/usr/bin/env python3
"""
Log Consumer with AI Classification and Metrics Push
Consumes logs from Kafka, classifies them, and pushes metrics to VictoriaMetrics
"""

import json
import time
import logging
import requests
from datetime import datetime
from kafka import KafkaConsumer
from kafka.errors import KafkaError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogClassifier:
    """Simple rule-based log classifier for demo purposes"""
    
    def __init__(self):
        # Keywords that indicate incidents (errors/critical issues)
        self.incident_keywords = [
            'down', 'crash', 'failed', 'error', 'timeout', 'unavailable',
            'memory', 'disk full', 'breach', 'overload', 'unreachable'
        ]
        
        # Keywords that indicate warnings
        self.warning_keywords = [
            'high', 'slow', 'low', 'degrading', 'approaching', 'growing',
            'nearly', 'expires', 'retry'
        ]
    
    def classify_log(self, log_entry):
        """
        Classify log entry into categories
        Returns: 'incident', 'warning', or 'normal'
        """
        level = log_entry.get('level', '').upper()
        message = log_entry.get('message', '').lower()
        
        # Rule-based classification
        if level in ['ERROR', 'CRITICAL']:
            return 'incident'
        elif level == 'WARN':
            return 'warning'
        elif level == 'INFO':
            # Check for incident indicators in INFO messages
            if any(keyword in message for keyword in self.incident_keywords):
                return 'incident'
            elif any(keyword in message for keyword in self.warning_keywords):
                return 'warning'
        
        return 'normal'

class MetricsPusher:
    """Push metrics to VictoriaMetrics"""
    
    def __init__(self, victoria_metrics_url='http://localhost:8428'):
        self.base_url = victoria_metrics_url
        self.session = requests.Session()
        
        # Initialize counters
        self.incident_count = 0
        self.warning_count = 0
        self.total_logs_count = 0
    
    def push_metric(self, metric_name, value, labels=None):
        """Push a single metric to VictoriaMetrics"""
        if labels is None:
            labels = {}
        
        # Format metric in Prometheus format
        labels_str = ','.join([f'{k}="{v}"' for k, v in labels.items()])
        if labels_str:
            metric_line = f'{metric_name}{{{labels_str}}} {value}'
        else:
            metric_line = f'{metric_name} {value}'
        
        try:
            url = f"{self.base_url}/api/v1/import/prometheus"
            headers = {'Content-Type': 'text/plain'}
            
            response = self.session.post(url, data=metric_line, headers=headers)
            response.raise_for_status()
            
            logger.debug(f"Successfully pushed metric: {metric_line}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to push metric {metric_name}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error pushing metric {metric_name}: {e}")
    
    def increment_incident_counter(self, service=None):
        """Increment incident counter"""
        self.incident_count += 1
        labels = {'type': 'incident'}
        if service:
            labels['service'] = service
        
        self.push_metric('log_incident_total', self.incident_count, labels)
        logger.info(f"Incident detected! Total incidents: {self.incident_count}")
    
    def increment_warning_counter(self, service=None):
        """Increment warning counter"""
        self.warning_count += 1
        labels = {'type': 'warning'}
        if service:
            labels['service'] = service
        
        self.push_metric('log_warning_total', self.warning_count, labels)
        logger.info(f"Warning detected! Total warnings: {self.warning_count}")
    
    def increment_total_logs_counter(self, log_level, service=None):
        """Increment total logs counter"""
        self.total_logs_count += 1
        labels = {'level': log_level.lower()}
        if service:
            labels['service'] = service
        
        self.push_metric('log_total', self.total_logs_count, labels)
    
    def push_processing_metrics(self, processing_time, classification):
        """Push processing time metrics"""
        labels = {'classification': classification}
        self.push_metric('log_processing_time_seconds', processing_time, labels)

class LogMonitorConsumer:
    """Main log monitoring consumer"""
    
    def __init__(self, 
                 kafka_bootstrap_servers='localhost:9092',
                 kafka_topic='logs',
                 victoria_metrics_url='http://localhost:8428',
                 consumer_group='log-monitor'):
        
        self.topic = kafka_topic
        self.classifier = LogClassifier()
        self.metrics_pusher = MetricsPusher(victoria_metrics_url)
        
        # Initialize Kafka consumer
        self.consumer = KafkaConsumer(
            kafka_topic,
            bootstrap_servers=kafka_bootstrap_servers,
            group_id=consumer_group,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            auto_offset_reset='latest',  # Start from latest messages
            enable_auto_commit=True,
            consumer_timeout_ms=1000  # Timeout for polling
        )
        
        logger.info(f"Initialized consumer for topic '{kafka_topic}' with group '{consumer_group}'")
    
    def process_log_entry(self, log_entry):
        """Process a single log entry"""
        start_time = time.time()
        
        try:
            # Extract log information
            timestamp = log_entry.get('timestamp', '')
            level = log_entry.get('level', 'UNKNOWN')
            message = log_entry.get('message', '')
            service = log_entry.get('service', 'unknown')
            
            # Classify the log
            classification = self.classifier.classify_log(log_entry)
            
            # Update metrics based on classification
            self.metrics_pusher.increment_total_logs_counter(level, service)
            
            if classification == 'incident':
                self.metrics_pusher.increment_incident_counter(service)
                logger.warning(f"🚨 INCIDENT: [{service}] {message}")
                
            elif classification == 'warning':
                self.metrics_pusher.increment_warning_counter(service)
                logger.warning(f"⚠️  WARNING: [{service}] {message}")
                
            else:
                logger.info(f"✅ NORMAL: [{service}] {message[:50]}...")
            
            # Push processing time metric
            processing_time = time.time() - start_time
            self.metrics_pusher.push_processing_metrics(processing_time, classification)
            
        except Exception as e:
            logger.error(f"Error processing log entry: {e}")
            logger.error(f"Log entry: {log_entry}")
    
    def run(self):
        """Run the consumer loop"""
        logger.info("Starting log monitoring consumer...")
        
        try:
            while True:
                try:
                    # Poll for messages
                    message_batch = self.consumer.poll(timeout_ms=1000)
                    
                    if message_batch:
                        for topic_partition, messages in message_batch.items():
                            for message in messages:
                                self.process_log_entry(message.value)
                    
                    # Small delay to prevent excessive CPU usage
                    time.sleep(0.1)
                    
                except KafkaError as e:
                    logger.error(f"Kafka error: {e}")
                    time.sleep(5)  # Wait before retrying
                    
                except Exception as e:
                    logger.error(f"Unexpected error in consumer loop: {e}")
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            logger.info("Shutting down consumer...")
        finally:
            self.consumer.close()
    
    def get_stats(self):
        """Get current statistics"""
        return {
            'total_incidents': self.metrics_pusher.incident_count,
            'total_warnings': self.metrics_pusher.warning_count,
            'total_logs': self.metrics_pusher.total_logs_count
        }

def main():
    """Main function to run the log consumer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Log monitoring consumer with AI classification')
    parser.add_argument('--kafka-servers', default='localhost:9092',
                       help='Kafka bootstrap servers (default: localhost:9092)')
    parser.add_argument('--kafka-topic', default='logs',
                       help='Kafka topic to consume from (default: logs)')
    parser.add_argument('--victoria-metrics-url', default='http://localhost:8428',
                       help='VictoriaMetrics URL (default: http://localhost:8428)')
    parser.add_argument('--consumer-group', default='log-monitor',
                       help='Kafka consumer group (default: log-monitor)')
    parser.add_argument('--stats-interval', type=int, default=30,
                       help='Interval to print stats in seconds (default: 30)')
    
    args = parser.parse_args()
    
    consumer = LogMonitorConsumer(
        kafka_bootstrap_servers=args.kafka_servers,
        kafka_topic=args.kafka_topic,
        victoria_metrics_url=args.victoria_metrics_url,
        consumer_group=args.consumer_group
    )
    
    # Start stats reporting in background (simple approach)
    import threading
    
    def print_stats():
        while True:
            time.sleep(args.stats_interval)
            stats = consumer.get_stats()
            logger.info(f"📊 STATS - Incidents: {stats['total_incidents']}, "
                       f"Warnings: {stats['total_warnings']}, "
                       f"Total Logs: {stats['total_logs']}")
    
    stats_thread = threading.Thread(target=print_stats, daemon=True)
    stats_thread.start()
    
    # Run the consumer
    consumer.run()

if __name__ == "__main__":
    main()
