#!/usr/bin/env python3
"""
Fake Log Producer for AI-driven Log Monitoring Demo
Generates various types of logs and sends them to Kafka
"""

import json
import time
import random
import logging
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogGenerator:
    def __init__(self, kafka_bootstrap_servers='localhost:9092', topic='logs'):
        """Initialize the log generator with Kafka connection"""
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        
        # Log templates for different scenarios
        self.log_templates = {
            'INFO': [
                'User login successful for user {}',
                'Database query executed in {} ms',
                'API request processed successfully',
                'Cache hit for key {}',
                'Health check passed',
                'Backup completed successfully',
                'Configuration reloaded',
                'Service started on port {}',
                'Scheduled task completed',
                'File uploaded successfully: {}'
            ],
            'WARN': [
                'High memory usage detected: {}% used',
                'Slow database response: {} ms',
                'Disk space low: {}% remaining',
                'Connection pool nearly exhausted: {} active connections',
                'API rate limit approaching: {} requests/min',
                'Queue size growing: {} items pending',
                'Cache miss ratio high: {}%',
                'Response time degrading: {} ms average',
                'Retry attempt {} for failed operation',
                'SSL certificate expires in {} days'
            ],
            'ERROR': [
                'Database connection failed: Connection timeout',
                'API endpoint returned 500 error',
                'Failed to process payment for order {}',
                'Authentication service unavailable',
                'File not found: {}',
                'Invalid JSON in request payload',
                'Network timeout connecting to external service',
                'Failed to send email notification',
                'Configuration file corrupted',
                'Insufficient permissions for operation'
            ],
            'CRITICAL': [
                'Database server is down - all queries failing',
                'Out of memory - service crashing',
                'Primary service crashed with exit code {}',
                'Disk full - cannot write logs',
                'Security breach detected - unauthorized access',
                'Load balancer health check failing',
                'All cache nodes unreachable',
                'Message queue broker down',
                'Critical configuration missing',
                'System overload - dropping connections'
            ]
        }
    
    def generate_log_entry(self):
        """Generate a single log entry with realistic data"""
        log_level = random.choices(
            ['INFO', 'WARN', 'ERROR', 'CRITICAL'],
            weights=[70, 20, 8, 2]  # Most logs are INFO, few are CRITICAL
        )[0]
        
        template = random.choice(self.log_templates[log_level])
        
        # Fill in template placeholders with realistic values
        if '{}' in template:
            if 'memory usage' in template:
                value = random.randint(75, 95)
            elif 'ms' in template:
                value = random.randint(500, 5000)
            elif 'remaining' in template:
                value = random.randint(5, 15)
            elif 'connections' in template:
                value = random.randint(90, 100)
            elif 'requests/min' in template:
                value = random.randint(4500, 5000)
            elif 'items pending' in template:
                value = random.randint(500, 1000)
            elif 'user' in template:
                value = f"user_{random.randint(1000, 9999)}"
            elif 'order' in template:
                value = f"ORD-{random.randint(10000, 99999)}"
            elif 'port' in template:
                value = random.choice([8080, 8443, 3000, 5432, 6379])
            elif 'exit code' in template:
                value = random.choice([1, 2, 137, 143])
            elif 'days' in template:
                value = random.randint(1, 30)
            elif 'File' in template or 'file' in template:
                value = f"/tmp/data_{random.randint(1, 1000)}.json"
            else:
                value = random.randint(1, 100)
            
            message = template.format(value)
        else:
            message = template
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': log_level,
            'message': message,
            'service': random.choice(['web-api', 'database', 'auth-service', 'payment', 'notification']),
            'host': f"host-{random.randint(1, 10)}",
            'request_id': f"req-{random.randint(100000, 999999)}"
        }
        
        return log_entry
    
    def send_log_batch(self, count=1):
        """Send a batch of log entries to Kafka"""
        for _ in range(count):
            log_entry = self.generate_log_entry()
            try:
                # Use log level as key for partitioning
                future = self.producer.send(
                    self.topic, 
                    value=log_entry, 
                    key=log_entry['level']
                )
                
                # Optional: wait for confirmation (comment out for higher throughput)
                # result = future.get(timeout=10)
                
                logger.info(f"Sent {log_entry['level']} log: {log_entry['message'][:50]}...")
                
            except KafkaError as e:
                logger.error(f"Failed to send log to Kafka: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
    
    def run_continuous(self, interval=2, batch_size=1):
        """Run continuous log generation"""
        logger.info(f"Starting continuous log generation (interval={interval}s, batch_size={batch_size})")
        
        try:
            while True:
                self.send_log_batch(batch_size)
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Stopping log generation...")
        except Exception as e:
            logger.error(f"Error in continuous generation: {e}")
        finally:
            self.producer.close()
    
    def simulate_incident(self, duration=30):
        """Simulate an incident with high error rates"""
        logger.info(f"Simulating incident for {duration} seconds...")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            # Generate more ERROR/CRITICAL logs during incident
            log_level = random.choices(
                ['INFO', 'WARN', 'ERROR', 'CRITICAL'],
                weights=[30, 20, 35, 15]  # Higher error rates
            )[0]
            
            template = random.choice(self.log_templates[log_level])
            
            # Similar logic as generate_log_entry but with forced error levels
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': log_level,
                'message': template.format(random.randint(1, 100)) if '{}' in template else template,
                'service': random.choice(['database', 'payment', 'auth-service']),
                'host': f"host-{random.randint(1, 3)}",  # Concentrate errors on fewer hosts
                'request_id': f"req-{random.randint(100000, 999999)}"
            }
            
            try:
                self.producer.send(self.topic, value=log_entry, key=log_entry['level'])
                logger.warning(f"INCIDENT LOG - {log_entry['level']}: {log_entry['message'][:50]}...")
            except Exception as e:
                logger.error(f"Failed to send incident log: {e}")
            
            time.sleep(0.5)  # Faster rate during incident
    
    def close(self):
        """Close the producer connection"""
        self.producer.close()

def main():
    """Main function to run the log generator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate fake logs for monitoring demo')
    parser.add_argument('--kafka-servers', default='localhost:9092', 
                       help='Kafka bootstrap servers (default: localhost:9092)')
    parser.add_argument('--topic', default='logs', 
                       help='Kafka topic name (default: logs)')
    parser.add_argument('--interval', type=float, default=2.0,
                       help='Interval between log batches in seconds (default: 2.0)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Number of logs per batch (default: 1)')
    parser.add_argument('--simulate-incident', action='store_true',
                       help='Simulate an incident scenario')
    parser.add_argument('--incident-duration', type=int, default=30,
                       help='Duration of incident simulation in seconds (default: 30)')
    
    args = parser.parse_args()
    
    generator = LogGenerator(args.kafka_servers, args.topic)
    
    try:
        if args.simulate_incident:
            generator.simulate_incident(args.incident_duration)
        else:
            generator.run_continuous(args.interval, args.batch_size)
    except KeyboardInterrupt:
        logger.info("Shutting down log generator...")
    finally:
        generator.close()

if __name__ == "__main__":
    main()
