"""
API Testing Examples
Shows how to use the REST API
"""

import requests
import json

# Base URL (change if running on different host/port)
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())

def test_list_models():
    """List available models"""
    response = requests.get(f"{BASE_URL}/models")
    print("\nAvailable Models:", json.dumps(response.json(), indent=2))

def test_single_prediction():
    """Test single prediction"""
    data = {
        "text": "This product is amazing! I love it!",
        "model": "3-class"
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print("\nSingle Prediction:", json.dumps(response.json(), indent=2))

def test_batch_prediction():
    """Test batch predictions"""
    data = {
        "texts": [
            "This is great!",
            "Not good at all.",
            "It's okay, nothing special."
        ],
        "model": "3-class"
    }
    response = requests.post(f"{BASE_URL}/batch", json=data)
    print("\nBatch Predictions:", json.dumps(response.json(), indent=2))

def test_model_comparison():
    """Test model comparison"""
    params = {"text": "I absolutely love this product!"}
    response = requests.post(f"{BASE_URL}/compare", params=params)
    print("\nModel Comparison:", json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Sentiment Analysis API")
    print("=" * 60)
    
    try:
        test_health()
        test_list_models()
        test_single_prediction()
        test_batch_prediction()
        test_model_comparison()
        print("\n✅ All tests passed!")
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("Make sure the API is running: python api.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
