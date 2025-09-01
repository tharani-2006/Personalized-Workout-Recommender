"""
Quick test to verify the API function works locally
Run this to test before deploying to Vercel
"""

import sys
import os
sys.path.append('api')

# Import the handler
from predict import handler

# Create a mock request for testing
class MockRequest:
    def __init__(self, data):
        self.data = data
        self.headers = {'Content-Length': str(len(data))}
    
    def read(self, length):
        return self.data

# Test the API function
def test_api():
    print("🧪 Testing API function locally...")
    
    # Create test data
    test_prompt = "I'm a 25-year-old male, 180cm, 75kg, want muscle gain, beginner"
    test_data = f'{{"prompt": "{test_prompt}"}}'.encode('utf-8')
    
    # Create handler instance
    api_handler = handler()
    
    # Mock the request
    api_handler.rfile = MockRequest(test_data)
    api_handler.headers = {'Content-Length': str(len(test_data))}
    
    # Test preprocessing
    try:
        features = api_handler.preprocess_prompt(test_prompt)
        print(f"✅ Preprocessing successful: {features}")
        
        # Test model
        model = api_handler.load_model()
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)
        
        print(f"✅ Prediction: {prediction}")
        print(f"✅ Probabilities: {probabilities}")
        
        print("🎉 API function works perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == '__main__':
    if test_api():
        print("\n🚀 Ready for Vercel deployment!")
    else:
        print("\n⚠️ Fix issues before deploying.")
