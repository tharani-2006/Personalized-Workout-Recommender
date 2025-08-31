"""
Test script to verify the complete workout recommender system works correctly.

Run this script to test all components before deployment.
"""

import requests
import json
import time
import sys
import os

def test_data_preprocessing():
    """Test the data preprocessing pipeline"""
    print("🧪 Testing Data Preprocessing Pipeline...")
    
    try:
        # Change to src directory
        os.chdir('src')
        
        # Import and test preprocessing
        from data_preprocessing import preprocess_data
        
        # Test with sample data
        X_train, X_test, y_train, y_test = preprocess_data('../data/train_sample.csv')
        
        print(f"✅ Preprocessing successful!")
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        print(f"   Features: {len(X_train.columns)}")
        print(f"   Classes: {len(set(y_train))}")
        
        # Go back to main directory
        os.chdir('..')
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {str(e)}")
        os.chdir('..')
        return False

def test_model_training():
    """Test model training and saving"""
    print("\n🤖 Testing Model Training...")
    
    try:
        # Check if model file exists
        if os.path.exists('models/workout_model.pkl'):
            print("✅ Model file found!")
            return True
        else:
            print("⚠️ Model file not found. Run 'python src/model.py' first.")
            return False
            
    except Exception as e:
        print(f"❌ Model check failed: {str(e)}")
        return False

def test_api_server():
    """Test if the Flask API server is running"""
    print("\n🌐 Testing API Server...")
    
    try:
        # Test health endpoint
        response = requests.get('http://localhost:5000/health', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API server is running!")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            return True
        else:
            print(f"❌ API server returned status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running. Start it with 'python app.py'")
        return False
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        return False

def test_prediction_api():
    """Test the prediction endpoint with sample data"""
    print("\n🎯 Testing Prediction API...")
    
    test_prompts = [
        {
            "name": "Muscle Building Test",
            "prompt": "I am a 25-year-old male with height of 180 cm and weight of 75 kg. My goal is muscle gain and I am a beginner at the gym.",
            "expected": "Muscle Building"
        },
        {
            "name": "Endurance Training Test", 
            "prompt": "I'm a 28-year-old woman, 165 cm and 60 kg. I want to lose weight and improve cardio. I'm intermediate at the gym.",
            "expected": "Endurance Training"
        },
        {
            "name": "Balanced Fitness Test",
            "prompt": "I'm 35, male, 175 cm and 70 kg. I want overall health and fitness. I'm intermediate level.",
            "expected": "Balanced Fitness"
        }
    ]
    
    success_count = 0
    
    for test_case in test_prompts:
        try:
            print(f"\n   Testing: {test_case['name']}")
            
            response = requests.post(
                'http://localhost:5000/predict',
                json={'prompt': test_case['prompt']},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                prediction = data['prediction']['workout_type']
                confidence = data['prediction']['confidence']
                
                print(f"   ✅ Prediction: {prediction} ({confidence}% confidence)")
                success_count += 1
            else:
                print(f"   ❌ API returned status: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Test failed: {str(e)}")
    
    print(f"\n📊 Prediction Tests: {success_count}/{len(test_prompts)} passed")
    return success_count == len(test_prompts)

def main():
    """Run all tests"""
    print("🧪" + "="*50)
    print("🚀 TESTING WORKOUT RECOMMENDER SYSTEM")
    print("🧪" + "="*50)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Data Preprocessing
    if test_data_preprocessing():
        tests_passed += 1
    
    # Test 2: Model Training
    if test_model_training():
        tests_passed += 1
    
    # Test 3: API Server
    if test_api_server():
        tests_passed += 1
        
        # Test 4: Prediction API (only if server is running)
        if test_prediction_api():
            tests_passed += 1
    else:
        print("\n⚠️ Skipping prediction tests (API server not running)")
        total_tests = 3
    
    # Final Results
    print("\n" + "🏆" + "="*50)
    print(f"📊 FINAL TEST RESULTS: {tests_passed}/{total_tests} PASSED")
    print("🏆" + "="*50)
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! Your system is ready for deployment! 🚀")
        print("\n📋 Next Steps:")
        print("   1. Deploy to Vercel using VERCEL_DEPLOYMENT_CHECKLIST.md")
        print("   2. Share your live URL with friends!")
        print("   3. Add to your resume/portfolio!")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\n💡 Common fixes:")
        print("   - Run 'python src/model.py' to train the model")
        print("   - Run 'python app.py' to start the API server")
        print("   - Check that all files are in the correct locations")

if __name__ == '__main__':
    main()
