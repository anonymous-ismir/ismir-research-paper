"""
Quick test script for the API server
Run this after starting the server to test endpoints
"""

import requests
import base64
import json
import time

API_BASE_URL = "http://localhost:8000"
# No API key required

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_BASE_URL}/api/v1/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_decide_cues():
    """Test decide cues endpoint"""
    print("Testing decide-cues endpoint...")
    url = f"{API_BASE_URL}/api/v1/decide-cues"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "story_text": "Suddenly rain started so i ran to shelter where i heard loud dog barking",
        "speed_wps": 2.0
    }
    
    response = requests.post(url, json=data, headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Found {len(result['cues'])} cues")
        print(f"Total duration: {result['total_duration_ms']}ms")
        print(f"First cue: {result['cues'][0] if result['cues'] else 'None'}")
    else:
        print(f"Error: {response.text}")
    print()

def test_generate_from_story():
    """Test complete pipeline"""
    print("Testing generate-from-story endpoint...")
    url = f"{API_BASE_URL}/api/v1/generate-from-story"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "story_text": "A dog barked loudly",
        "speed_wps": 2.0,
        "return_cues": True
    }
    
    print("Sending request (this may take a while for audio generation)...")
    start_time = time.time()
    response = requests.post(url, json=data, headers=headers, timeout=300)
    elapsed = time.time() - start_time
    
    print(f"Status: {response.status_code}")
    print(f"Time taken: {elapsed:.2f} seconds")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Generated audio: {result['duration_ms']}ms")
        print(f"Audio cues: {len(result.get('cues', []))}")
        print(f"Audio base64 length: {len(result['audio_base64'])} characters")
        
        # Save audio
        audio_bytes = base64.b64decode(result["audio_base64"])
        output_file = "test_output.wav"
        with open(output_file, "wb") as f:
            f.write(audio_bytes)
        print(f"Audio saved to {output_file}")
    else:
        print(f"Error: {response.text}")
    print()

# No authentication test needed - API is open

if __name__ == "__main__":
    print("=" * 50)
    print("Audio Generation API Test Suite")
    print("=" * 50)
    print()
    
    # Wait for server to be ready
    print("Waiting for server to be ready...")
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=2)
            if response.status_code == 200:
                print("Server is ready!")
                break
        except requests.exceptions.RequestException:
            if i < max_retries - 1:
                print(f"Waiting... ({i+1}/{max_retries})")
                time.sleep(2)
            else:
                print("ERROR: Server is not responding. Make sure the server is running.")
                exit(1)
    
    print()
    
    # Run tests
    test_health()
    test_decide_cues()
    # No authentication test needed - API is open
    
    # Uncomment to test full audio generation (takes longer)
    # test_generate_from_story()
    
    print("=" * 50)
    print("Tests completed!")
    print("=" * 50)

