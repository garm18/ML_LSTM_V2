import requests
import json

def test_api():
    # URL lokal
    base_url = 'http://127.0.0.1:5000'
    
    # Test 1: Prediction Endpoint
    print("\nTesting Prediction Endpoint:")
    prediction_url = f"{base_url}/predict"
    test_data = {
        "features": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    
    response = requests.post(
        prediction_url,
        headers={'Content-Type': 'application/json'},
        json=test_data
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    # Test 2: Raw Data Endpoint
    print("\nTesting Raw Data Endpoint:")
    raw_data_url = f"{base_url}/raw-data"
    response = requests.get(raw_data_url)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Raw data retrieved successfully")
    
    # Test 3: Data Summary Endpoint
    print("\nTesting Data Summary Endpoint:")
    summary_url = f"{base_url}/data-summary"
    response = requests.get(summary_url)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Summary data retrieved successfully")

if __name__ == "__main__":
    test_api()