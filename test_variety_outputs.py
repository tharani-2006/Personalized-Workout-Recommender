"""
Test script to demonstrate the variety and sophistication of workout recommendations.
Shows that the system gives different outputs based on user characteristics.
"""

import requests
import json
import time

def test_prediction_variety():
    """Test various user profiles to show prediction variety"""
    
    test_cases = [
        {
            "name": "Young Male - Muscle Building",
            "prompt": "I am a 22-year-old male with height of 185 cm and weight of 70 kg. My goal is muscle gain and I am a beginner at the gym.",
            "expected_type": "Muscle Building"
        },
        {
            "name": "Young Female - Weight Loss", 
            "prompt": "I'm a 24-year-old woman, 160 cm tall and weigh 65 kg. I want to lose weight and improve my cardio fitness. I'm intermediate at the gym.",
            "expected_type": "Endurance Training"
        },
        {
            "name": "Middle-aged Beginner",
            "prompt": "I am a 45-year-old male with height of 175 cm and weight of 85 kg. My goal is health maintenance and I am a beginner at the gym.",
            "expected_type": "Balanced Fitness"
        },
        {
            "name": "Advanced Female - Strength",
            "prompt": "I'm a 30-year-old woman, 168 cm and 58 kg. I want to build muscle and get stronger. I'm advanced at the gym with years of experience.",
            "expected_type": "Muscle Building"
        },
        {
            "name": "Overweight Male - Health",
            "prompt": "I am a 35-year-old male with height of 180 cm and weight of 95 kg. My goal is weight loss and overall health. I'm a beginner.",
            "expected_type": "Endurance Training"
        },
        {
            "name": "Young Female - Endurance",
            "prompt": "I'm a 26-year-old woman, 165 cm and 55 kg. I want to improve my endurance for running marathons. I'm intermediate level.",
            "expected_type": "Endurance Training"
        },
        {
            "name": "Older Male - General Fitness",
            "prompt": "I am a 50-year-old male with height of 172 cm and weight of 78 kg. My goal is staying healthy and active. I'm intermediate at the gym.",
            "expected_type": "Balanced Fitness"
        },
        {
            "name": "Young Male - Lean Build",
            "prompt": "I'm a 23-year-old guy, 178 cm tall and weigh 62 kg. I want to build muscle and gain weight. I'm a beginner at the gym.",
            "expected_type": "Muscle Building"
        }
    ]
    
    print("🧪" + "="*70)
    print("🎯 TESTING WORKOUT RECOMMENDATION VARIETY")
    print("🧪" + "="*70)
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print(f"👤 Profile: {test_case['prompt'][:80]}...")
        
        try:
            response = requests.post(
                'http://localhost:5000/predict',
                json={'prompt': test_case['prompt']},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                prediction = data['prediction']
                
                workout_type = prediction['workout_type']
                confidence = prediction['confidence']
                probabilities = prediction['probabilities']
                
                print(f"🎯 Prediction: {workout_type} ({confidence}% confidence)")
                print(f"📊 Breakdown: {probabilities}")
                
                # Check if prediction matches expectation
                matches_expected = workout_type == test_case['expected_type']
                status = "✅ Expected" if matches_expected else "🔄 Different"
                print(f"   {status} (Expected: {test_case['expected_type']})")
                
                results.append({
                    'test': test_case['name'],
                    'prediction': workout_type,
                    'confidence': confidence,
                    'expected': test_case['expected_type'],
                    'matches': matches_expected,
                    'probabilities': probabilities
                })
                
            else:
                print(f"❌ API Error: {response.status_code}")
                results.append({
                    'test': test_case['name'],
                    'error': f"HTTP {response.status_code}"
                })
                
        except Exception as e:
            print(f"❌ Request failed: {str(e)}")
            results.append({
                'test': test_case['name'],
                'error': str(e)
            })
        
        time.sleep(0.5)  # Small delay between requests
    
    # Summary
    print("\n" + "🏆" + "="*70)
    print("📊 PREDICTION VARIETY ANALYSIS")
    print("🏆" + "="*70)
    
    successful_tests = [r for r in results if 'error' not in r]
    
    if successful_tests:
        # Count prediction types
        prediction_counts = {}
        confidence_scores = []
        
        for result in successful_tests:
            pred_type = result['prediction']
            prediction_counts[pred_type] = prediction_counts.get(pred_type, 0) + 1
            confidence_scores.append(result['confidence'])
        
        print(f"\n📈 Prediction Distribution:")
        for workout_type, count in prediction_counts.items():
            percentage = (count / len(successful_tests)) * 100
            print(f"   {workout_type}: {count}/{len(successful_tests)} ({percentage:.1f}%)")
        
        print(f"\n🎯 Confidence Statistics:")
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        min_confidence = min(confidence_scores)
        max_confidence = max(confidence_scores)
        print(f"   Average: {avg_confidence:.1f}%")
        print(f"   Range: {min_confidence:.1f}% - {max_confidence:.1f}%")
        
        variety_score = len(prediction_counts)
        print(f"\n🌟 Variety Score: {variety_score}/3 workout types predicted")
        
        if variety_score == 3:
            print("🎉 EXCELLENT! System shows great variety and sophistication!")
        elif variety_score == 2:
            print("✅ GOOD! System shows reasonable variety.")
        else:
            print("⚠️ Limited variety - consider enhancing the classification logic.")
            
    else:
        print("❌ No successful predictions to analyze.")
    
    return results

if __name__ == '__main__':
    print("🚀 Make sure Flask server is running at http://localhost:5000")
    print("   Run: python app.py")
    print("\nStarting variety test in 3 seconds...")
    time.sleep(3)
    
    results = test_prediction_variety()
    
    print(f"\n🎯 Test completed! {len(results)} test cases processed.")
    print("💡 This demonstrates the sophistication of your ML system!")
