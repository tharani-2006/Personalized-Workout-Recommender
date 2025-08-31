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
expected_features = None

# Human-friendly workout type mapping
class_names = ['cardio', 'mixed', 'strength']  # Technical names for model
human_friendly_names = {
    'cardio': 'Endurance Training',
    'mixed': 'Balanced Fitness',
    'strength': 'Muscle Building'
}

def load_production_model():
    """
    Load the trained model and feature names for making predictions.

    Returns:
        tuple: (sklearn model, list of expected feature names)
    """
    model_path = 'models/workout_model.pkl'
    features_path = 'models/feature_names.pkl'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature names file not found at {features_path}. Please train the model first.")

    # Load model
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)

    # Load expected feature names
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)

    print(f"✅ Model loaded successfully from {model_path}")
    print(f"📋 Feature names loaded: {len(feature_names)} features")

    return loaded_model, feature_names

def preprocess_single_prompt(prompt_text):
    """
    Preprocess a single user prompt for prediction to match training data format.

    Args:
        prompt_text (str): User input prompt

    Returns:
        pd.DataFrame: Preprocessed features ready for model prediction
    """
    try:
        # Extract user characteristics
        user_features = extract_user_characteristics_from_prompt(prompt_text)

        # Create a dummy completion for feature extraction (we'll use average values)
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

        # Remove the primary_workout_type column (it's only used for target creation)
        if 'primary_workout_type' in df.columns:
            df = df.drop(['primary_workout_type'], axis=1)

        # One-hot encode categorical features to match training format
        categorical_columns = ['gender', 'goal', 'gym_level', 'workout_duration', 'workout_intensity']
        df_encoded = pd.get_dummies(df, columns=categorical_columns)

        # Use the global expected_features loaded from training
        global expected_features

        # Ensure all expected features are present
        for feature in expected_features:
            if feature not in df_encoded.columns:
                df_encoded[feature] = 0

        # Keep only expected features in the correct order
        df_final = df_encoded[expected_features]

        # Normalize numerical features (using same ranges as training)
        from sklearn.preprocessing import MinMaxScaler
        numerical_features = ['age', 'height', 'weight', 'total_exercises', 'total_volume',
                             'rest_days', 'workout_days', 'avg_reps_per_exercise', 'exercise_variety']

        # Apply approximate normalization (in production, save the actual scaler)
        for feature in numerical_features:
            if feature in df_final.columns:
                # Use approximate min/max values from training data
                feature_ranges = {
                    'age': (18, 65), 'height': (150, 200), 'weight': (45, 120),
                    'total_exercises': (0, 20), 'total_volume': (0, 600),
                    'rest_days': (0, 7), 'workout_days': (0, 7),
                    'avg_reps_per_exercise': (0, 25), 'exercise_variety': (0, 15)
                }

                if feature in feature_ranges:
                    min_val, max_val = feature_ranges[feature]
                    df_final[feature] = (df_final[feature] - min_val) / (max_val - min_val)
                    df_final[feature] = df_final[feature].clip(0, 1)  # Ensure values are in [0,1]

        return df_final

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
        
        # Convert numeric prediction to human-friendly name
        technical_workout_type = class_names[prediction_numeric]
        human_workout_type = human_friendly_names[technical_workout_type]
        confidence = float(max(prediction_probabilities))

        # Prepare response with human-friendly names
        response = {
            'status': 'success',
            'prediction': {
                'workout_type': human_workout_type,
                'technical_type': technical_workout_type,
                'confidence': round(confidence * 100, 2),
                'probabilities': {
                    human_friendly_names[class_names[i]]: round(float(prob) * 100, 2)
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
        # Load the trained model and feature names
        print("🚀 Starting Workout Recommender API...")
        model, expected_features = load_production_model()

        print("🌐 Starting Flask server...")
        print("📍 API will be available at: http://localhost:5000")
        print("🔗 Prediction endpoint: http://localhost:5000/predict")
        print("❤️ Health check: http://localhost:5000/health")

        # Start the Flask development server
        app.run(host='0.0.0.0', port=5000, debug=True)

    except Exception as e:
        print(f"❌ Failed to start API: {str(e)}")
        print("💡 Make sure you've trained the model first by running: python src/model.py")
