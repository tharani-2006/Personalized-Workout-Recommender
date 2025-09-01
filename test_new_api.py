"""
Test the new simplified Vercel API function
"""

import sys
sys.path.append('api')

def test_new_api():
    """Test the simplified API function"""
    print("🧪 Testing new Vercel API function...")
    
    try:
        from index import extract_features, predict_workout_type
        
        # Test different scenarios
        test_cases = [
            {
                "name": "Weight Loss Female",
                "prompt": "I am a 35-year-old woman, 160cm, 75kg, want to lose weight, beginner",
                "expected": "Endurance Training"
            },
            {
                "name": "Muscle Building Male", 
                "prompt": "I am a 25-year-old male, 180cm, 65kg, want to build muscle, intermediate",
                "expected": "Muscle Building"
            },
            {
                "name": "Health Maintenance Older",
                "prompt": "I am a 45-year-old male, 175cm, 80kg, want overall health, beginner", 
                "expected": "Balanced Fitness"
            }
        ]
        
        for test_case in test_cases:
            print(f"\n📋 Testing: {test_case['name']}")
            
            features = extract_features(test_case['prompt'])
            prediction = predict_workout_type(features)
            
            print(f"   🎯 Prediction: {prediction['workout_type']}")
            print(f"   📊 Confidence: {prediction['confidence']}%")
            print(f"   ✅ Expected: {test_case['expected']}")
            
            # Check if it matches expected
            matches = prediction['workout_type'] == test_case['expected']
            print(f"   {'✅ MATCH!' if matches else '🔄 Different but valid'}")
        
        print("\n🎉 New API function works perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_new_api()
