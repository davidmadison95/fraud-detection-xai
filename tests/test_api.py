"""
API Testing Script
Test the Flask fraud detection API endpoints.
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:5000"


def test_home():
    """Test home endpoint."""
    print("="*60)
    print("Testing HOME endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    print()


def test_health():
    """Test health check endpoint."""
    print("="*60)
    print("Testing HEALTH endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    print()


def test_model_info():
    """Test model info endpoint."""
    print("="*60)
    print("Testing MODEL INFO endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/model_info")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    print()


def test_score_single():
    """Test single transaction scoring."""
    print("="*60)
    print("Testing SCORE endpoint (Single Transaction)")
    print("="*60)
    
    # High-risk transaction
    transaction = {
        "transaction_id": "TXN_TEST_001",
        "amount_usd": 2500.00,
        "merchant_category": "electronics",
        "merchant_country": "CN",
        "channel": "online",
        "card_present": 0,
        "customer_age_days": 45,
        "customer_txn_30d": 2,
        "avg_amount_30d": 75.50,
        "std_amount_30d": 25.30,
        "country_mismatch": 1,
        "hour_of_day": 3,
        "is_weekend": 0
    }
    
    print("Request payload:")
    print(json.dumps(transaction, indent=2))
    print()
    
    response = requests.post(
        f"{BASE_URL}/score",
        json=transaction,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    print()


def test_score_legitimate():
    """Test scoring a legitimate transaction."""
    print("="*60)
    print("Testing SCORE endpoint (Legitimate Transaction)")
    print("="*60)
    
    # Low-risk transaction
    transaction = {
        "transaction_id": "TXN_TEST_002",
        "amount_usd": 45.99,
        "merchant_category": "groceries",
        "merchant_country": "US",
        "channel": "pos",
        "card_present": 1,
        "customer_age_days": 850,
        "customer_txn_30d": 15,
        "avg_amount_30d": 52.30,
        "std_amount_30d": 18.50,
        "country_mismatch": 0,
        "hour_of_day": 14,
        "is_weekend": 0
    }
    
    print("Request payload:")
    print(json.dumps(transaction, indent=2))
    print()
    
    response = requests.post(
        f"{BASE_URL}/score",
        json=transaction,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    print()


def test_batch_score():
    """Test batch scoring."""
    print("="*60)
    print("Testing BATCH SCORE endpoint")
    print("="*60)
    
    # Multiple transactions
    payload = {
        "records": [
            {
                "transaction_id": "TXN_BATCH_001",
                "amount_usd": 3000.00,
                "merchant_category": "electronics",
                "merchant_country": "RU",
                "channel": "online",
                "card_present": 0,
                "customer_age_days": 30,
                "customer_txn_30d": 1,
                "avg_amount_30d": 50.00,
                "std_amount_30d": 15.00,
                "country_mismatch": 1,
                "hour_of_day": 2,
                "is_weekend": 1
            },
            {
                "transaction_id": "TXN_BATCH_002",
                "amount_usd": 25.50,
                "merchant_category": "restaurants",
                "merchant_country": "US",
                "channel": "pos",
                "card_present": 1,
                "customer_age_days": 500,
                "customer_txn_30d": 20,
                "avg_amount_30d": 30.00,
                "std_amount_30d": 10.00,
                "country_mismatch": 0,
                "hour_of_day": 12,
                "is_weekend": 0
            },
            {
                "transaction_id": "TXN_BATCH_003",
                "amount_usd": 1500.00,
                "merchant_category": "travel",
                "merchant_country": "UK",
                "channel": "online",
                "card_present": 0,
                "customer_age_days": 120,
                "customer_txn_30d": 5,
                "avg_amount_30d": 200.00,
                "std_amount_30d": 100.00,
                "country_mismatch": 0,
                "hour_of_day": 15,
                "is_weekend": 0
            }
        ]
    }
    
    print(f"Request payload: {len(payload['records'])} transactions")
    print()
    
    response = requests.post(
        f"{BASE_URL}/batch_score",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    print()


def test_error_handling():
    """Test error handling with invalid data."""
    print("="*60)
    print("Testing ERROR HANDLING")
    print("="*60)
    
    # Missing required field
    invalid_transaction = {
        "transaction_id": "TXN_INVALID",
        "amount_usd": 100.00
        # Missing many required fields
    }
    
    print("Sending incomplete data:")
    print(json.dumps(invalid_transaction, indent=2))
    print()
    
    response = requests.post(
        f"{BASE_URL}/score",
        json=invalid_transaction,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    print()


def run_all_tests():
    """Run all API tests."""
    print("\n" + "="*60)
    print("FRAUD DETECTION API TEST SUITE")
    print("="*60 + "\n")
    
    try:
        # Test basic endpoints
        test_home()
        test_health()
        test_model_info()
        
        # Test scoring endpoints
        test_score_single()
        test_score_legitimate()
        test_batch_score()
        
        # Test error handling
        test_error_handling()
        
        print("="*60)
        print("✓ ALL TESTS COMPLETED")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n" + "="*60)
        print("ERROR: Could not connect to API")
        print("="*60)
        print("\nPlease make sure the API server is running:")
        print("  python serve_api.py")
        print("="*60)
    
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")


if __name__ == "__main__":
    run_all_tests()
