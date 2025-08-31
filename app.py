"""
Personalized Workout Recommender - Flask API

This Flask application provides a REST API endpoint for workout type predictions.
Users can submit their characteristics and receive personalized workout recommendations.

Author: ML Team
Date: 2025-08-31
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os
import sys
import traceback

# Add src directory to path for imports
sys.path.append('src')
from data_preprocessing import extract_user_characteristics_from_prompt, extract_workout_features

# Initialize Flask application
app = Flask(__name__)

# Global variables for model and preprocessing components
model = None
class_names = ['cardio', 'mixed', 'strength']

def load_production_model():
    """
    Load the trained model for making predictions.
    
    Returns:
        sklearn model: Loaded Random Forest model
    """
    model_path = 'models/workout_model.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    print(f"✅ Model loaded successfully from {model_path}")
    return loaded_model

def preprocess_single_prompt(prompt_text):
    """
    Preprocess a single user prompt for prediction.
    
    Args:
        prompt_text (str): User input prompt
        
    Returns:
        pd.DataFrame: Preprocessed features ready for model prediction
    """
    try:
        # Extract user characteristics
        user_features = extract_user_characteristics_from_prompt(prompt_text)
        
        # Create a dummy completion for feature extraction (we'll use average values)
        # In a real scenario, we'd need to handle this differently
        dummy_completion = """Monday:
- Push-ups: 3 sets of 12 reps
- Running: 3 sets of 15 reps
Tuesday:
- Rest Day
Wednesday:
- Squats: 3 sets of 10 reps
Thursday:
- Rest Day
Friday:
- Cycling: 3 sets of 20 reps
Saturday:
- Rest Day
Sunday:
- Rest Day"""
        
        # Extract workout features (using dummy data for now)
        workout_features = extract_workout_features(dummy_completion)
        
        # Combine all features
        all_features = {**user_features, **workout_features}
        
        # Create DataFrame
        df = pd.DataFrame([all_features])
        
        # Handle missing values with defaults
        df = df.fillna({
            'age': 30, 'height': 170, 'weight': 70,
            'gender': 'male', 'goal': 'health maintenance', 'gym_level': 'beginner'
        })
        
        return df
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

@app.route('/')
def home():
    """
    Serve the main page with the workout recommendation form.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_workout_type():
    """
    API endpoint for workout type prediction.
    
    Expected JSON input:
    {
        "prompt": "I am a 25-year-old male with height of 180 cm and weight of 75 kg..."
    }
    
    Returns:
        JSON response with prediction and confidence
    """
    try:
        # Get JSON data from request
        request_data = request.get_json(force=True)
        
        if 'prompt' not in request_data:
            return jsonify({
                'error': 'Missing required field: prompt',
                'status': 'error'
            }), 400
        
        user_prompt = request_data['prompt']
        
        # Preprocess the input
        processed_data = preprocess_single_prompt(user_prompt)
        
        # Make prediction
        prediction_numeric = model.predict(processed_data)[0]
        prediction_probabilities = model.predict_proba(processed_data)[0]
        
        # Convert numeric prediction to class name
        predicted_workout_type = class_names[prediction_numeric]
        confidence = float(max(prediction_probabilities))
        
        # Prepare response
        response = {
            'status': 'success',
            'prediction': {
                'workout_type': predicted_workout_type,
                'confidence': round(confidence * 100, 2),
                'probabilities': {
                    class_names[i]: round(float(prob) * 100, 2) 
                    for i, prob in enumerate(prediction_probabilities)
                }
            },
            'user_input': user_prompt
        }
        
        return jsonify(response)
        
    except Exception as e:
        # Handle errors gracefully
        error_response = {
            'status': 'error',
            'error': str(e),
            'message': 'Failed to process workout recommendation request'
        }
        print(f"API Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify(error_response), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify API is running.
    """
    return jsonify({
        'status': 'healthy',
        'message': 'Workout Recommender API is running',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    try:
        # Load the trained model
        print("🚀 Starting Workout Recommender API...")
        model = load_production_model()
        
        print("🌐 Starting Flask server...")
        print("📍 API will be available at: http://localhost:5000")
        print("🔗 Prediction endpoint: http://localhost:5000/predict")
        print("❤️ Health check: http://localhost:5000/health")
        
        # Start the Flask development server
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        print(f"❌ Failed to start API: {str(e)}")
        print("💡 Make sure you've trained the model first by running: python src/model.py")
