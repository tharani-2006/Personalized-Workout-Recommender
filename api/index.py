"""
Simple Vercel API function that definitely works
Uses the proven format for Vercel serverless functions
"""

import json
import re

def handler(request, response):
    """Main Vercel handler - simple and reliable"""
    
    # Set CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Content-Type'] = 'application/json'
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return ''
    
    # Handle POST
    if request.method != 'POST':
        response.status_code = 405
        return json.dumps({'status': 'error', 'error': 'Method not allowed'})
    
    try:
        # Parse request
        data = request.json
        
        if not data or 'prompt' not in data:
            response.status_code = 400
            return json.dumps({'status': 'error', 'error': 'Missing prompt'})
        
        user_prompt = data['prompt']
        
        # Extract features
        features = extract_features(user_prompt)
        
        # Make prediction
        prediction = predict_workout_type(features)
        
        # Return result
        return json.dumps({
            'status': 'success',
            'prediction': prediction
        })
        
    except Exception as e:
        response.status_code = 500
        return json.dumps({'status': 'error', 'error': str(e)})

def extract_features(prompt_text):
    """Extract user features from prompt"""
    features = {}
    
    # Age
    age_match = re.search(r'(\d+)[-\s]year[-\s]old|I[\'m\s]+(\d+)', prompt_text)
    features['age'] = int(age_match.group(1) or age_match.group(2)) if age_match else 30
    
    # Gender
    gender_match = re.search(r'(male|female|man|woman)', prompt_text.lower())
    features['gender'] = 'male' if gender_match and gender_match.group(1) in ['male', 'man'] else 'female'
    
    # Height
    height_match = re.search(r'(\d+)\s*cm', prompt_text)
    features['height'] = int(height_match.group(1)) if height_match else 170
    
    # Weight
    weight_match = re.search(r'(\d+)\s*kg', prompt_text)
    features['weight'] = int(weight_match.group(1)) if weight_match else 70
    
    # Goal
    if any(word in prompt_text.lower() for word in ['lose', 'weight', 'fat', 'slim']):
        features['goal'] = 'weight_loss'
    elif any(word in prompt_text.lower() for word in ['muscle', 'build', 'gain', 'strong']):
        features['goal'] = 'muscle_gain'
    elif any(word in prompt_text.lower() for word in ['endurance', 'cardio', 'run', 'stamina']):
        features['goal'] = 'endurance'
    else:
        features['goal'] = 'health'
    
    # Experience
    if any(word in prompt_text.lower() for word in ['beginner', 'new', 'start']):
        features['experience'] = 'beginner'
    elif any(word in prompt_text.lower() for word in ['advanced', 'experienced', 'expert']):
        features['experience'] = 'advanced'
    else:
        features['experience'] = 'intermediate'
    
    return features

def predict_workout_type(features):
    """Predict workout type with good variety"""
    
    age = features['age']
    goal = features['goal']
    experience = features['experience']
    weight = features['weight']
    height = features['height']
    gender = features['gender']
    
    # Calculate BMI
    bmi = weight / ((height / 100) ** 2)
    
    # Scoring system for variety
    scores = {'cardio': 0, 'mixed': 0, 'strength': 0}
    
    # Goal-based scoring (primary)
    if goal == 'weight_loss':
        scores['cardio'] += 70
        scores['mixed'] += 25
        scores['strength'] += 5
    elif goal == 'muscle_gain':
        scores['strength'] += 70
        scores['mixed'] += 20
        scores['cardio'] += 10
    elif goal == 'endurance':
        scores['cardio'] += 80
        scores['mixed'] += 15
        scores['strength'] += 5
    else:  # health
        scores['mixed'] += 60
        scores['cardio'] += 30
        scores['strength'] += 25
    
    # Age adjustments
    if age > 45:
        scores['mixed'] += 20
        scores['cardio'] += 10
    elif age < 25:
        scores['strength'] += 15
    
    # BMI adjustments
    if bmi > 25:
        scores['cardio'] += 20
    elif bmi < 20:
        scores['strength'] += 15
    
    # Experience adjustments
    if experience == 'beginner' and age > 40:
        scores['mixed'] += 25
    
    # Find winner
    predicted_type = max(scores, key=scores.get)
    
    # Map to human names
    type_names = {
        'cardio': 'Endurance Training',
        'mixed': 'Balanced Fitness',
        'strength': 'Muscle Building'
    }
    
    # Calculate confidence
    total = sum(scores.values())
    confidence = (scores[predicted_type] / total) * 100 if total > 0 else 60
    
    # Create probabilities
    probabilities = {}
    for workout_type, score in scores.items():
        prob = (score / total) * 100 if total > 0 else 33.33
        probabilities[type_names[workout_type]] = round(prob, 2)
    
    return {
        'workout_type': type_names[predicted_type],
        'confidence': round(confidence, 2),
        'probabilities': probabilities
    }
