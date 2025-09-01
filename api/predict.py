"""
Vercel Serverless Function for Workout Prediction
Simple format that works without vercel.json configuration
"""

import json
import re
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Handle POST requests for workout predictions"""
        try:
            # Read request data
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            if 'prompt' not in data:
                self.send_error_response('Missing required field: prompt', 400)
                return

            user_prompt = data['prompt']

            # Extract features and make prediction
            features = extract_features(user_prompt)
            prediction = predict_workout_type(features)

            # Send successful response
            response = {
                'status': 'success',
                'prediction': prediction
            }

            self.send_success_response(response)

        except Exception as e:
            self.send_error_response(f'Prediction failed: {str(e)}', 500)

    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def send_success_response(self, data):
        """Send successful JSON response with CORS headers"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def send_error_response(self, message, status_code):
        """Send error JSON response with CORS headers"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        error_data = {'status': 'error', 'error': message}
        self.wfile.write(json.dumps(error_data).encode('utf-8'))

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
