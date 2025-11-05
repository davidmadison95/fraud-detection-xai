"""
Flask REST API for Fraud Detection Scoring
Provides real-time fraud scoring endpoint.
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FraudFeatureEngineer

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and preprocessor
MODEL = None
PREPROCESSOR = None
FEATURE_ENGINEER = None


def load_artifacts():
    """Load model and preprocessor at startup."""
    global MODEL, PREPROCESSOR, FEATURE_ENGINEER
    
    try:
        MODEL = joblib.load('../models/fraud_model.pkl')
        PREPROCESSOR = joblib.load('../models/preprocessor.pkl')
        
        FEATURE_ENGINEER = FraudFeatureEngineer()
        FEATURE_ENGINEER.preprocessor = PREPROCESSOR
        FEATURE_ENGINEER._extract_feature_names()
        
        print("✅ Model and preprocessor loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model artifacts: {e}")
        return False


@app.route('/')
def home():
    """API home endpoint with documentation."""
    return jsonify({
        'service': 'Financial Fraud Detection API',
        'version': '1.0.0',
        'status': 'active',
        'endpoints': {
            '/health': 'Health check endpoint',
            '/score': 'Score transactions for fraud (POST)',
            '/batch_score': 'Batch score multiple transactions (POST)'
        },
        'documentation': {
            '/score': {
                'method': 'POST',
                'description': 'Score a single transaction',
                'payload_example': {
                    'transaction_id': 'TXN_00000001',
                    'timestamp': '2023-06-15 14:30:00',
                    'amount_usd': 542.10,
                    'merchant_category': 'electronics',
                    'merchant_country': 'US',
                    'channel': 'online',
                    'card_present': 0,
                    'customer_id': 1234567,
                    'customer_age_days': 450,
                    'customer_txn_30d': 12,
                    'avg_amount_30d': 85.50,
                    'std_amount_30d': 42.30,
                    'country_mismatch': 0,
                    'hour_of_day': 14,
                    'is_weekend': 0
                }
            }
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    model_loaded = MODEL is not None and PREPROCESSOR is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded
    })


@app.route('/score', methods=['POST'])
def score():
    """
    Score a single transaction for fraud probability.
    
    Expected JSON payload:
    {
        "transaction_id": "TXN_00000001",
        "timestamp": "2023-06-15 14:30:00",
        "amount_usd": 542.10,
        "merchant_category": "electronics",
        "merchant_country": "US",
        "channel": "online",
        "card_present": 0,
        "customer_id": 1234567,
        "customer_age_days": 450,
        "customer_txn_30d": 12,
        "avg_amount_30d": 85.50,
        "std_amount_30d": 42.30,
        "country_mismatch": 0,
        "hour_of_day": 14,
        "is_weekend": 0
    }
    """
    try:
        # Parse request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Validate required fields
        required_fields = [
            'amount_usd', 'merchant_category', 'merchant_country', 'channel',
            'card_present', 'customer_id', 'customer_age_days', 'customer_txn_30d',
            'avg_amount_30d', 'std_amount_30d', 'country_mismatch',
            'hour_of_day', 'is_weekend'
        ]
        
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Transform features
        X = FEATURE_ENGINEER.transform(df)
        
        # Get prediction
        fraud_probability = float(MODEL.predict_proba(X)[0, 1])
        fraud_prediction = int(fraud_probability >= 0.5)
        
        # Determine risk level
        if fraud_probability >= 0.7:
            risk_level = 'HIGH'
        elif fraud_probability >= 0.3:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # Return response
        response = {
            'transaction_id': data.get('transaction_id', 'N/A'),
            'fraud_probability': round(fraud_probability, 4),
            'fraud_prediction': fraud_prediction,
            'risk_level': risk_level,
            'threshold': 0.5,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Scoring failed',
            'message': str(e)
        }), 500


@app.route('/batch_score', methods=['POST'])
def batch_score():
    """
    Score multiple transactions in batch.
    
    Expected JSON payload:
    {
        "records": [
            {
                "transaction_id": "TXN_00000001",
                "amount_usd": 542.10,
                ...
            },
            {
                "transaction_id": "TXN_00000002",
                "amount_usd": 125.50,
                ...
            }
        ]
    }
    """
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'records' not in data:
            return jsonify({'error': 'No records provided'}), 400
        
        records = data['records']
        
        if not isinstance(records, list) or len(records) == 0:
            return jsonify({'error': 'Records must be a non-empty list'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Validate required fields
        required_fields = [
            'amount_usd', 'merchant_category', 'merchant_country', 'channel',
            'card_present', 'customer_id', 'customer_age_days', 'customer_txn_30d',
            'avg_amount_30d', 'std_amount_30d', 'country_mismatch',
            'hour_of_day', 'is_weekend'
        ]
        
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Transform features
        X = FEATURE_ENGINEER.transform(df)
        
        # Get predictions
        fraud_probabilities = MODEL.predict_proba(X)[:, 1]
        fraud_predictions = (fraud_probabilities >= 0.5).astype(int)
        
        # Build response
        results = []
        for i, (prob, pred) in enumerate(zip(fraud_probabilities, fraud_predictions)):
            if prob >= 0.7:
                risk_level = 'HIGH'
            elif prob >= 0.3:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            results.append({
                'transaction_id': records[i].get('transaction_id', f'txn_{i}'),
                'fraud_probability': round(float(prob), 4),
                'fraud_prediction': int(pred),
                'risk_level': risk_level
            })
        
        response = {
            'total_transactions': len(results),
            'flagged_count': sum(r['fraud_prediction'] for r in results),
            'results': results,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Batch scoring failed',
            'message': str(e)
        }), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Return model information and performance metrics."""
    try:
        # Try to load metrics if available
        metrics_path = '../models/fraud_model_metrics.json'
        metrics = {}
        
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        
        info = {
            'model_type': 'XGBoost Classifier',
            'model_loaded': MODEL is not None,
            'features_count': len(FEATURE_ENGINEER.get_feature_names()) if FEATURE_ENGINEER else 0,
            'performance_metrics': metrics,
            'version': '1.0.0'
        }
        
        return jsonify(info), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to retrieve model info',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/', '/health', '/score', '/batch_score', '/model_info']
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500


if __name__ == '__main__':
    print("🚀 Starting Fraud Detection API...")
    
    # Load model artifacts
    if not load_artifacts():
        print("⚠️  Warning: Model artifacts not loaded. API may not function correctly.")
    
    # Run Flask app
    print("📡 API server running on http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  - GET  /         : API documentation")
    print("  - GET  /health   : Health check")
    print("  - POST /score    : Score single transaction")
    print("  - POST /batch_score : Batch score transactions")
    print("  - GET  /model_info : Model information")
    print("\nPress CTRL+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
