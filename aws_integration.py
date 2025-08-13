# 🔧 AWS Integration Configuration

import boto3
import json
from datetime import datetime, timedelta
import os

class AWSLogMonitor:
    """Enhanced log monitor with AWS CloudWatch integration"""
    
    def __init__(self, region='us-east-1'):
        self.region = region
        self.cloudwatch_logs = boto3.client('logs', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.sns = boto3.client('sns', region_name=region)
        
    def get_log_groups(self):
        """Discover available log groups in AWS"""
        try:
            response = self.cloudwatch_logs.describe_log_groups(limit=50)
            return [lg['logGroupName'] for lg in response['logGroups']]
        except Exception as e:
            print(f"Error fetching log groups: {e}")
            return []
    
    def stream_cloudwatch_logs(self, log_group_name, hours_back=1):
        """Stream logs from AWS CloudWatch"""
        try:
            # Get logs from last N hours
            start_time = int((datetime.now() - timedelta(hours=hours_back)).timestamp() * 1000)
            end_time = int(datetime.now().timestamp() * 1000)
            
            response = self.cloudwatch_logs.filter_log_events(
                logGroupName=log_group_name,
                startTime=start_time,
                endTime=end_time,
                limit=100
            )
            
            logs = []
            for event in response['events']:
                log_entry = {
                    'timestamp': datetime.fromtimestamp(event['timestamp'] / 1000).isoformat(),
                    'message': event['message'],
                    'log_stream': event.get('logStreamName', ''),
                    'source': 'aws_cloudwatch',
                    'log_group': log_group_name
                }
                logs.append(log_entry)
            
            return logs
            
        except Exception as e:
            print(f"Error streaming CloudWatch logs: {e}")
            return []
    
    def get_ec2_logs(self):
        """Get common EC2 instance logs"""
        common_log_groups = [
            '/aws/ec2/system',
            '/aws/ec2/application', 
            '/var/log/messages',
            '/var/log/secure',
            '/var/log/httpd/access_log',
            '/var/log/httpd/error_log'
        ]
        
        all_logs = []
        available_groups = self.get_log_groups()
        
        for log_group in common_log_groups:
            if log_group in available_groups:
                logs = self.stream_cloudwatch_logs(log_group)
                all_logs.extend(logs)
        
        return all_logs
    
    def get_lambda_logs(self):
        """Get AWS Lambda function logs"""
        lambda_groups = [lg for lg in self.get_log_groups() if lg.startswith('/aws/lambda/')]
        
        all_logs = []
        for log_group in lambda_groups[:5]:  # Limit to first 5 functions
            logs = self.stream_cloudwatch_logs(log_group)
            all_logs.extend(logs)
        
        return all_logs
    
    def get_api_gateway_logs(self):
        """Get API Gateway logs"""
        api_groups = [lg for lg in self.get_log_groups() if 'API-Gateway' in lg]
        
        all_logs = []
        for log_group in api_groups:
            logs = self.stream_cloudwatch_logs(log_group)
            all_logs.extend(logs)
        
        return all_logs
    
    def get_all_aws_logs(self):
        """Comprehensive AWS log collection"""
        print("🔍 Collecting AWS logs...")
        
        all_logs = []
        
        # Collect different types of logs
        print("  📊 Getting EC2 logs...")
        all_logs.extend(self.get_ec2_logs())
        
        print("  ⚡ Getting Lambda logs...")  
        all_logs.extend(self.get_lambda_logs())
        
        print("  🌐 Getting API Gateway logs...")
        all_logs.extend(self.get_api_gateway_logs())
        
        print(f"✅ Collected {len(all_logs)} AWS log entries")
        return all_logs

# Usage example
if __name__ == "__main__":
    # Initialize AWS monitor
    aws_monitor = AWSLogMonitor(region='us-east-1')  # Change to your region
    
    # Discover available log groups
    print("Available AWS Log Groups:")
    log_groups = aws_monitor.get_log_groups()
    for lg in log_groups[:10]:  # Show first 10
        print(f"  - {lg}")
    
    # Collect logs from various AWS services
    aws_logs = aws_monitor.get_all_aws_logs()
    
    # Output sample logs
    print(f"\n📋 Sample AWS Logs ({len(aws_logs)} total):")
    for log in aws_logs[:3]:
        print(f"  [{log['timestamp']}] {log['log_group']}: {log['message'][:100]}...")
