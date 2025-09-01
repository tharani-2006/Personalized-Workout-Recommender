"""
Test the Vercel API function locally to ensure it works before deployment
"""

import sys
import json
sys.path.append('api')

def test_vercel_api():
    """Test the Vercel API function format"""
    print("🧪 Testing Vercel API function...")
    
    try:
        # Import the handler
        from predict import handler, preprocess_prompt_advanced, load_model
        
        print("✅ Import successful")
        
        # Test preprocessing
        test_prompt = "I am a 35-year-old woman, 160cm, 75kg, want to lose weight, beginner"
        features = preprocess_prompt_advanced(test_prompt)
        print(f"✅ Preprocessing: {features}")
        
        # Test model
        model = load_model()
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)
        print(f"✅ Prediction: {prediction}")
        print(f"✅ Probabilities: {probabilities}")
        
        print("🎉 Vercel API function is ready!")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    if test_vercel_api():
        print("\n🚀 API function works! Ready for Vercel deployment.")
    else:
        print("\n⚠️ Fix API issues before deploying to Vercel.")
