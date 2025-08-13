# 🤖 Enhanced ML Model Training for Log Classification

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedLogClassifier:
    """Enhanced ML model for log classification with multiple algorithms"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.feature_importance = None
        
    def generate_comprehensive_training_data(self, num_samples=5000):
        """Generate comprehensive synthetic training data"""
        logger.info(f"Generating {num_samples} synthetic log samples...")
        
        logs = []
        labels = []
        
        # Enhanced error/incident templates
        incident_templates = [
            # Database issues
            "ERROR: Database connection timeout after {timeout}s - Connection pool exhausted",
            "CRITICAL: Database deadlock detected in table {table} - Transaction rolled back",
            "ERROR: SQL query failed with syntax error: {query}",
            "FATAL: Database server {server} is unreachable - All connections failed",
            
            # Application crashes
            "ERROR: OutOfMemoryError in {service} - Heap space exhausted",
            "CRITICAL: {service} crashed with exit code {code} - Core dump generated",
            "ERROR: NullPointerException in {module}.{function}() at line {line}",
            "FATAL: Segmentation fault in {process} - Process terminated unexpectedly",
            
            # Network/Infrastructure
            "ERROR: Connection refused to {host}:{port} - Service unavailable",
            "CRITICAL: Load balancer {lb} failed health check - Removing from rotation",
            "ERROR: SSL certificate expired for {domain} - HTTPS connections failing",
            "FATAL: Disk full on {mount} - Write operations failing",
            
            # Security incidents  
            "CRITICAL: Unauthorized access attempt from IP {ip} - Account locked",
            "ERROR: Authentication failed for user {user} - Invalid credentials",
            "CRITICAL: SQL injection attempt detected in parameter {param}",
            "ERROR: Rate limit exceeded from {ip} - {requests} requests in {time}s",
            
            # Performance issues
            "ERROR: API timeout - Response time {time}ms exceeds threshold {max}ms",
            "CRITICAL: CPU usage {usage}% on {server} - Performance degraded",
            "ERROR: Memory leak detected in {service} - Usage increased {percent}%",
            "FATAL: Thread pool exhausted - {active}/{max} threads in use"
        ]
        
        # Enhanced warning templates
        warning_templates = [
            # Performance warnings
            "WARNING: High response time {time}ms for endpoint {endpoint}",
            "WARNING: Memory usage {usage}% on {server} approaching limit",
            "WARNING: CPU usage {usage}% on {server} - Monitor closely",
            "WARNING: Disk usage {usage}% on {mount} - Cleanup recommended",
            
            # Application warnings
            "WARNING: Deprecated API {api} used by {client} - Update required",
            "WARNING: Cache hit ratio {ratio}% below optimal threshold",
            "WARNING: Connection pool {pool} at {usage}% capacity",
            "WARNING: Queue depth {depth} exceeding normal range",
            
            # Infrastructure warnings
            "WARNING: Load average {load} on {server} above normal",
            "WARNING: Network latency {latency}ms to {service} elevated",
            "WARNING: Retry attempt {attempt}/{max} for {operation}",
            "WARNING: Backup job for {database} took {duration} minutes",
            
            # Security warnings
            "WARNING: Multiple failed login attempts for user {user}",
            "WARNING: Unusual traffic pattern detected from {country}",
            "WARNING: Certificate {cert} expires in {days} days",
            "WARNING: Firewall rule {rule} blocked {count} attempts"
        ]
        
        # Normal operation templates
        normal_templates = [
            # Successful operations
            "INFO: User {user} logged in successfully from {ip}",
            "INFO: API request to {endpoint} completed in {time}ms",
            "INFO: Database query executed successfully - {rows} rows affected",
            "INFO: File {file} uploaded successfully - Size: {size}MB",
            
            # System operations
            "INFO: Scheduled backup completed for {database} - Duration: {time}",
            "INFO: Cache refreshed for {service} - {entries} entries updated",
            "INFO: Health check passed for {service} - Status: OK",
            "INFO: Log rotation completed for {logfile}",
            
            # Business events
            "INFO: Order {order_id} processed successfully for customer {customer}",
            "INFO: Payment of ${amount} authorized for transaction {tx_id}",
            "INFO: Email notification sent to {email} - Template: {template}",
            "INFO: Report {report} generated successfully - {pages} pages"
        ]
        
        # Generate samples with realistic distributions
        incident_count = int(num_samples * 0.05)  # 5% incidents
        warning_count = int(num_samples * 0.25)   # 25% warnings
        normal_count = num_samples - incident_count - warning_count  # 70% normal
        
        # Generate incidents
        for _ in range(incident_count):
            template = np.random.choice(incident_templates)
            log_message = self._fill_template(template)
            logs.append(log_message)
            labels.append("incident")
        
        # Generate warnings
        for _ in range(warning_count):
            template = np.random.choice(warning_templates)
            log_message = self._fill_template(template)
            logs.append(log_message)
            labels.append("warning")
            
        # Generate normal logs
        for _ in range(normal_count):
            template = np.random.choice(normal_templates)
            log_message = self._fill_template(template)
            logs.append(log_message)
            labels.append("normal")
        
        # Create DataFrame and shuffle
        df = pd.DataFrame({"log_message": logs, "classification": labels})
        df = df.sample(frac=1).reset_index(drop=True)
        
        logger.info(f"Generated data distribution:")
        logger.info(f"  - Incidents: {incident_count} ({incident_count/num_samples*100:.1f}%)")
        logger.info(f"  - Warnings: {warning_count} ({warning_count/num_samples*100:.1f}%)")
        logger.info(f"  - Normal: {normal_count} ({normal_count/num_samples*100:.1f}%)")
        
        return df
    
    def _fill_template(self, template):
        """Fill template with realistic values"""
        replacements = {
            '{timeout}': str(np.random.choice([30, 60, 120, 300])),
            '{table}': np.random.choice(['users', 'orders', 'products', 'sessions', 'logs']),
            '{query}': 'SELECT * FROM ' + np.random.choice(['users', 'orders', 'products']),
            '{server}': np.random.choice(['db-01', 'web-02', 'app-03', 'cache-01']),
            '{service}': np.random.choice(['auth-service', 'payment-api', 'user-service', 'notification-service']),
            '{code}': str(np.random.choice([1, 2, 137, 139, 255])),
            '{module}': np.random.choice(['UserController', 'PaymentProcessor', 'DataValidator']),
            '{function}': np.random.choice(['processOrder', 'validateUser', 'sendNotification']),
            '{line}': str(np.random.randint(10, 999)),
            '{process}': np.random.choice(['java', 'python', 'node', 'nginx']),
            '{host}': np.random.choice(['api.example.com', '192.168.1.100', 'db.internal']),
            '{port}': str(np.random.choice([80, 443, 3306, 5432, 6379])),
            '{lb}': np.random.choice(['lb-01', 'lb-02', 'nginx-proxy']),
            '{domain}': np.random.choice(['api.company.com', 'app.company.com', 'secure.company.com']),
            '{mount}': np.random.choice(['/var', '/tmp', '/home', '/opt']),
            '{ip}': f"{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
            '{user}': np.random.choice(['admin', 'john.doe', 'api_user', 'guest']),
            '{param}': np.random.choice(['user_id', 'search_query', 'file_name']),
            '{requests}': str(np.random.randint(100, 1000)),
            '{time}': str(np.random.randint(1000, 10000)),
            '{max}': str(np.random.randint(500, 2000)),
            '{usage}': str(np.random.randint(80, 99)),
            '{percent}': str(np.random.randint(10, 50)),
            '{active}': str(np.random.randint(90, 100)),
            '{endpoint}': np.random.choice(['/api/users', '/api/orders', '/api/products']),
            '{client}': np.random.choice(['mobile-app', 'web-app', 'third-party']),
            '{ratio}': str(np.random.randint(60, 80)),
            '{pool}': np.random.choice(['db_pool', 'redis_pool', 'http_pool']),
            '{depth}': str(np.random.randint(50, 200)),
            '{load}': str(round(np.random.uniform(2.0, 5.0), 2)),
            '{latency}': str(np.random.randint(100, 500)),
            '{attempt}': str(np.random.randint(1, 5)),
            '{operation}': np.random.choice(['database_connect', 'api_call', 'file_write']),
            '{database}': np.random.choice(['userdb', 'orderdb', 'logdb']),
            '{duration}': str(np.random.randint(15, 120)),
            '{country}': np.random.choice(['China', 'Russia', 'Unknown']),
            '{cert}': np.random.choice(['ssl_cert', 'api_cert', 'root_ca']),
            '{days}': str(np.random.randint(1, 30)),
            '{rule}': np.random.choice(['BLOCK_SUSPICIOUS', 'RATE_LIMIT', 'GEO_BLOCK']),
            '{count}': str(np.random.randint(10, 100)),
            '{file}': np.random.choice(['document.pdf', 'image.jpg', 'data.csv']),
            '{size}': str(round(np.random.uniform(0.1, 100.0), 1)),
            '{entries}': str(np.random.randint(100, 10000)),
            '{logfile}': np.random.choice(['access.log', 'error.log', 'app.log']),
            '{order_id}': f"ORD-{np.random.randint(10000, 99999)}",
            '{customer}': f"CUST-{np.random.randint(1000, 9999)}",
            '{amount}': str(round(np.random.uniform(10.0, 1000.0), 2)),
            '{tx_id}': f"TX-{np.random.randint(100000, 999999)}",
            '{email}': f"user{np.random.randint(1, 100)}@company.com",
            '{template}': np.random.choice(['welcome', 'order_confirmation', 'password_reset']),
            '{report}': np.random.choice(['sales_report', 'user_analytics', 'system_health']),
            '{pages}': str(np.random.randint(1, 50))
        }
        
        # Replace placeholders
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, value)
        
        return template
    
    def train_multiple_models(self, df):
        """Train multiple ML models and compare performance"""
        logger.info("Training multiple ML models...")
        
        # Prepare data
        X = df['log_message']
        y = self.label_encoder.fit_transform(df['classification'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models and vectorizers
        vectorizers = {
            'tfidf': TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words='english'),
            'count': CountVectorizer(max_features=2000, ngram_range=(1, 2), stop_words='english')
        }
        
        models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'naive_bayes': MultinomialNB(),
            'svm': SVC(kernel='linear', probability=True, random_state=42)
        }
        
        results = {}
        best_score = 0
        
        # Train all combinations
        for vec_name, vectorizer in vectorizers.items():
            for model_name, model in models.items():
                logger.info(f"Training {model_name} with {vec_name}...")
                
                # Create pipeline
                pipeline = Pipeline([
                    ('vectorizer', vectorizer),
                    ('classifier', model)
                ])
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Cross-validation score
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1_macro')
                mean_cv_score = cv_scores.mean()
                
                # Test score
                test_score = pipeline.score(X_test, y_test)
                
                # Store results
                combination_name = f"{model_name}_{vec_name}"
                results[combination_name] = {
                    'pipeline': pipeline,
                    'cv_score': mean_cv_score,
                    'test_score': test_score,
                    'cv_std': cv_scores.std()
                }
                
                logger.info(f"  CV Score: {mean_cv_score:.4f} (±{cv_scores.std():.4f})")
                logger.info(f"  Test Score: {test_score:.4f}")
                
                # Track best model
                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    self.best_model = pipeline
                    self.best_model_name = combination_name
        
        # Store results
        self.training_results = results
        
        # Generate detailed evaluation
        self._evaluate_best_model(X_test, y_test)
        
        return results
    
    def _evaluate_best_model(self, X_test, y_test):
        """Detailed evaluation of the best model"""
        logger.info(f"Evaluating best model: {self.best_model_name}")
        
        # Predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        logger.info("Classification Report:")
        for class_name in class_names:
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            logger.info(f"  {class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        # Overall metrics
        logger.info(f"Overall Accuracy: {report['accuracy']:.3f}")
        logger.info(f"Macro F1-Score: {report['macro avg']['f1-score']:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Store evaluation metrics
        self.evaluation_metrics = {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names.tolist()
        }
    
    def hyperparameter_tuning(self, df):
        """Perform hyperparameter tuning on best model type"""
        logger.info("Starting hyperparameter tuning...")
        
        X = df['log_message']
        y = self.label_encoder.fit_transform(df['classification'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define parameter grid for Random Forest (typically performs well)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('rf', RandomForestClassifier(random_state=42))
        ])
        
        param_grid = {
            'tfidf__max_features': [1000, 2000, 3000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'rf__n_estimators': [50, 100, 200],
            'rf__max_depth': [10, 20, None],
            'rf__min_samples_split': [2, 5]
        }
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, scoring='f1_macro', 
            n_jobs=-1, verbose=1
        )
        
        logger.info("Running grid search (this may take a while)...")
        grid_search.fit(X_train, y_train)
        
        # Best model
        self.best_tuned_model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Test performance
        test_score = self.best_tuned_model.score(X_test, y_test)
        logger.info(f"Test score with tuned model: {test_score:.4f}")
        
        return grid_search.best_params_
    
    def save_models(self, model_dir="models"):
        """Save trained models and vectorizers"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        logger.info(f"Saving models to {model_dir}/...")
        
        # Save best model
        if self.best_model:
            model_path = os.path.join(model_dir, "best_log_classifier.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            logger.info(f"Saved best model: {model_path}")
        
        # Save tuned model if available
        if hasattr(self, 'best_tuned_model'):
            tuned_path = os.path.join(model_dir, "tuned_log_classifier.pkl")
            with open(tuned_path, 'wb') as f:
                pickle.dump(self.best_tuned_model, f)
            logger.info(f"Saved tuned model: {tuned_path}")
        
        # Save label encoder
        encoder_path = os.path.join(model_dir, "label_encoder.pkl")
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        logger.info(f"Saved label encoder: {encoder_path}")
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'best_model_name': getattr(self, 'best_model_name', 'unknown'),
            'evaluation_metrics': getattr(self, 'evaluation_metrics', {}),
            'class_names': self.label_encoder.classes_.tolist()
        }
        
        metadata_path = os.path.join(model_dir, "training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {metadata_path}")
    
    def load_model(self, model_path="models/best_log_classifier.pkl"):
        """Load trained model"""
        try:
            with open(model_path, 'rb') as f:
                self.best_model = pickle.load(f)
            
            # Load label encoder
            encoder_path = model_path.replace('best_log_classifier.pkl', 'label_encoder.pkl')
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_log(self, log_message):
        """Classify a single log message"""
        if not self.best_model:
            raise ValueError("No model loaded. Train or load a model first.")
        
        prediction = self.best_model.predict([log_message])[0]
        probabilities = self.best_model.predict_proba([log_message])[0]
        
        # Convert prediction back to class name
        class_name = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence scores
        class_names = self.label_encoder.classes_
        confidence_scores = dict(zip(class_names, probabilities))
        
        return {
            'classification': class_name,
            'confidence': max(probabilities),
            'all_probabilities': confidence_scores
        }

def main():
    """Main training pipeline"""
    logger.info("🚀 Starting advanced ML model training...")
    
    # Initialize classifier
    classifier = AdvancedLogClassifier()
    
    # Generate training data
    logger.info("📊 Generating comprehensive training dataset...")
    df = classifier.generate_comprehensive_training_data(num_samples=8000)
    
    # Display data statistics
    logger.info("📈 Dataset Statistics:")
    logger.info(f"Total samples: {len(df)}")
    logger.info("Class distribution:")
    for class_name, count in df['classification'].value_counts().items():
        percentage = count / len(df) * 100
        logger.info(f"  - {class_name}: {count} ({percentage:.1f}%)")
    
    # Train multiple models
    logger.info("🎯 Training and comparing multiple ML models...")
    results = classifier.train_multiple_models(df)
    
    # Show results summary
    logger.info("📊 Model Comparison Results:")
    for model_name, result in sorted(results.items(), key=lambda x: x[1]['cv_score'], reverse=True):
        cv_score = result['cv_score']
        test_score = result['test_score']
        logger.info(f"  {model_name}: CV={cv_score:.4f}, Test={test_score:.4f}")
    
    # Hyperparameter tuning
    logger.info("⚙️ Performing hyperparameter tuning...")
    best_params = classifier.hyperparameter_tuning(df)
    
    # Save models
    logger.info("💾 Saving trained models...")
    classifier.save_models()
    
    # Test the model with examples
    logger.info("🧪 Testing model with example logs...")
    test_logs = [
        "ERROR: Database connection failed - timeout after 30 seconds",
        "WARNING: High CPU usage 85% detected on web-01 server",
        "INFO: User john.doe logged in successfully from 192.168.1.100",
        "CRITICAL: OutOfMemoryError in payment-service - heap space exhausted",
        "WARNING: Disk usage 90% on /var partition - cleanup recommended"
    ]
    
    for test_log in test_logs:
        prediction = classifier.predict_log(test_log)
        logger.info(f"  Log: {test_log[:50]}...")
        logger.info(f"  → Classification: {prediction['classification']} (confidence: {prediction['confidence']:.3f})")
    
    logger.info("✅ Advanced ML model training completed successfully!")
    logger.info(f"📁 Models saved in './models/' directory")
    
    return classifier

if __name__ == "__main__":
    # Set up model directory
    os.makedirs("models", exist_ok=True)
    
    # Run training
    trained_classifier = main()
    
    print("\n🎯 Training Summary:")
    print("=" * 50)
    print(f"✅ Models trained and saved successfully")
    print(f"📊 Best model: {trained_classifier.best_model_name}")
    print(f"🎯 Ready for integration with real-time consumer")
    print(f"📁 Model files: ./models/")
    print("=" * 50)
