

import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def extract_user_characteristics_from_prompt(prompt_text):
    """
    Extract user demographic and fitness characteristics from prompt text.

    Args:
        prompt_text (str): User input text containing personal information

    Returns:
        dict: Dictionary containing extracted user characteristics

    Example:
        Input: "I am a 25-year-old male with height of 180 cm..."
        Output: {'age': 25, 'gender': 'male', 'height': 180, ...}
    """
    characteristics = {}

    # Extract age (e.g., "21-year-old" -> 21)
    age_match = re.search(r'(\d+)-year-old', prompt_text)
    characteristics['age'] = int(age_match.group(1)) if age_match else None

    # Extract gender (male/female)
    gender_match = re.search(r'(male|female)', prompt_text)
    characteristics['gender'] = gender_match.group(1) if gender_match else None

    # Extract height in cm (e.g., "height of 180 cm" -> 180)
    height_match = re.search(r'height of (\d+) cm', prompt_text)
    characteristics['height'] = int(height_match.group(1)) if height_match else None

    # Extract weight in kg (e.g., "weight of 75 kg" -> 75)
    weight_match = re.search(r'weight of (\d+) kg', prompt_text)
    characteristics['weight'] = int(weight_match.group(1)) if weight_match else None

    # Extract fitness goal (e.g., "goal is weight loss" -> "weight loss")
    goal_match = re.search(r'goal is (.*?)(?:, and|\.)', prompt_text)
    characteristics['goal'] = goal_match.group(1).strip() if goal_match else None

    # Extract gym experience level (beginner/intermediate/advanced)
    gym_level_match = re.search(r'I am a (beginner|intermediate|advanced) at the gym', prompt_text)
    characteristics['gym_level'] = gym_level_match.group(1) if gym_level_match else None

    return characteristics


def categorize_exercise_type(exercise_name):
    """
    Categorize an exercise as cardio, strength, or flexibility based on its name.

    Args:
        exercise_name (str): Name of the exercise

    Returns:
        str: Exercise category ('cardio', 'strength', 'flexibility')
    """
    # Define exercise categories based on common workout types
    cardio_exercises = {
        'running', 'cycling', 'walking', 'jump rope', 'jumping jacks',
        'high knees', 'mountain climbers', 'burpees'
    }

    strength_exercises = {
        'bench press', 'deadlifts', 'squats', 'pull-ups', 'push-ups',
        'dumbbell rows', 'shoulder press', 'bicep curls', 'tricep dips',
        'leg press', 'bodyweight squats'
    }

    flexibility_exercises = {
        'planks', 'sit-ups', 'russian twists'  # Core/stability exercises
    }

    # Normalize exercise name for comparison
    exercise_lower = exercise_name.lower().strip()

    # Check each category
    if any(cardio in exercise_lower for cardio in cardio_exercises):
        return 'cardio'
    elif any(strength in exercise_lower for strength in strength_exercises):
        return 'strength'
    elif any(flex in exercise_lower for flex in flexibility_exercises):
        return 'flexibility'
    else:
        return 'strength'  # Default to strength if unknown


def parse_exercise_data(completion_text):
    """
    Parse individual exercises from workout completion text.

    Args:
        completion_text (str): The workout plan text

    Returns:
        list: List of tuples (exercise_name, sets, reps)

    Example:
        Input: "- Push-ups: 3 sets of 15 reps\n- Running: 2 sets of 20 reps"
        Output: [('Push-ups', 3, 15), ('Running', 2, 20)]
    """
    exercise_pattern = r'- ([^:]+): (\d+) sets of (\d+) reps'
    raw_exercises = re.findall(exercise_pattern, completion_text)

    # Convert to proper data types and clean exercise names
    parsed_exercises = []
    for exercise_name, sets_str, reps_str in raw_exercises:
        exercise_name_clean = exercise_name.strip()
        sets_int = int(sets_str)
        reps_int = int(reps_str)
        parsed_exercises.append((exercise_name_clean, sets_int, reps_int))

    return parsed_exercises


def calculate_workout_volume_metrics(parsed_exercises):
    """
    Calculate volume-based workout metrics from parsed exercise data.

    Args:
        parsed_exercises (list): List of (exercise_name, sets, reps) tuples

    Returns:
        dict: Volume metrics including total_volume, avg_reps, etc.
    """
    if not parsed_exercises:
        return {
            'total_exercises': 0,
            'total_volume': 0,
            'avg_reps_per_exercise': 0,
            'exercise_variety': 0
        }

    # Calculate basic volume metrics
    total_exercises = len(parsed_exercises)
    total_volume = sum(sets * reps for _, sets, reps in parsed_exercises)
    all_reps = [reps for _, _, reps in parsed_exercises]
    avg_reps_per_exercise = np.mean(all_reps)

    # Calculate exercise variety (unique exercise count)
    unique_exercises = set(exercise_name for exercise_name, _, _ in parsed_exercises)
    exercise_variety = len(unique_exercises)

    return {
        'total_exercises': total_exercises,
        'total_volume': total_volume,
        'avg_reps_per_exercise': avg_reps_per_exercise,
        'exercise_variety': exercise_variety
    }


def categorize_workout_duration(total_volume):
    """
    Categorize workout duration based on total volume (sets × reps).

    Args:
        total_volume (int): Total workout volume

    Returns:
        str: Duration category ('short', 'medium', 'long')

    Categories:
        - Short: ≤200 total volume (light workout)
        - Medium: 201-400 total volume (moderate workout)
        - Long: >400 total volume (intense workout)
    """
    if total_volume <= 200:
        return 'short'
    elif total_volume <= 400:
        return 'medium'
    else:
        return 'long'


def categorize_workout_intensity(avg_reps_per_exercise):
    """
    Categorize workout intensity based on average reps per exercise.

    Args:
        avg_reps_per_exercise (float): Average repetitions per exercise

    Returns:
        str: Intensity category ('high', 'medium', 'low')

    Logic:
        - High intensity: ≤8 reps (heavy weights, explosive movements)
        - Medium intensity: 9-15 reps (moderate weights, balanced)
        - Low intensity: >15 reps (light weights, endurance focus)
    """
    if avg_reps_per_exercise <= 8:
        return 'high'
    elif avg_reps_per_exercise <= 15:
        return 'medium'
    else:
        return 'low'


def determine_primary_workout_type(parsed_exercises):
    """
    Determine the primary workout type based on exercise composition.

    Args:
        parsed_exercises (list): List of (exercise_name, sets, reps) tuples

    Returns:
        str: Primary workout type ('cardio', 'strength', 'mixed')

    Logic:
        - Cardio: Majority cardiovascular exercises
        - Strength: Majority resistance/weight training exercises
        - Mixed: Balanced combination of both types
    """
    if not parsed_exercises:
        return 'mixed'

    exercise_names = [exercise_name for exercise_name, _, _ in parsed_exercises]

    # Count exercises by type using our categorization function
    cardio_count = sum(1 for name in exercise_names
                      if categorize_exercise_type(name) == 'cardio')
    strength_count = sum(1 for name in exercise_names
                        if categorize_exercise_type(name) == 'strength')

    # Determine primary type based on counts
    if cardio_count > strength_count:
        return 'cardio'
    elif strength_count > cardio_count:
        return 'strength'
    else:
        return 'mixed'


def extract_workout_features(completion_text):
    """
    Extract comprehensive workout features from completion text using modular approach.

    Args:
        completion_text (str): The workout plan text

    Returns:
        dict: Dictionary containing all extracted workout features

    Features extracted:
        - Basic metrics: total_exercises, total_volume, exercise_variety
        - Schedule metrics: rest_days, workout_days
        - Intensity metrics: avg_reps_per_exercise, workout_intensity
        - Duration metrics: workout_duration
        - Type classification: primary_workout_type
    """
    # Step 1: Parse exercise data from text
    parsed_exercises = parse_exercise_data(completion_text)

    # Step 2: Calculate volume-based metrics
    volume_metrics = calculate_workout_volume_metrics(parsed_exercises)

    # Step 3: Count rest days in the 7-day plan
    rest_days_count = len(re.findall(r'Rest Day', completion_text))
    workout_days_count = 7 - rest_days_count

    # Step 4: Categorize duration and intensity
    workout_duration_category = categorize_workout_duration(volume_metrics['total_volume'])
    workout_intensity_category = categorize_workout_intensity(volume_metrics['avg_reps_per_exercise'])

    # Step 5: Determine primary workout type
    primary_workout_type = determine_primary_workout_type(parsed_exercises)

    # Step 6: Combine all features into final dictionary
    comprehensive_features = {
        # Volume and variety metrics
        'total_exercises': volume_metrics['total_exercises'],
        'total_volume': volume_metrics['total_volume'],
        'exercise_variety': volume_metrics['exercise_variety'],
        'avg_reps_per_exercise': volume_metrics['avg_reps_per_exercise'],

        # Schedule metrics
        'rest_days': rest_days_count,
        'workout_days': workout_days_count,

        # Categorized features
        'workout_duration': workout_duration_category,
        'workout_intensity': workout_intensity_category,
        'primary_workout_type': primary_workout_type
    }

    return comprehensive_features


def analyze_class_distribution(y, class_names=None):
    """
    Analyze and display class distribution to check for imbalance.

    Args:
        y: Target variable array
        class_names: Optional list of class names for display

    Returns:
        dict: Class distribution statistics
    """
    unique, counts = np.unique(y, return_counts=True)
    total_samples = len(y)

    print("\n" + "="*40)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*40)

    distribution = {}
    for class_val, count in zip(unique, counts):
        percentage = (count / total_samples) * 100
        class_name = class_names[class_val] if class_names else f"Class {class_val}"
        distribution[class_val] = {'count': count, 'percentage': percentage}
        print(f"{class_name}: {count} samples ({percentage:.1f}%)")

    # Check for significant imbalance (if any class has <20% or >60% of data)
    percentages = [dist['percentage'] for dist in distribution.values()]
    min_percent = min(percentages)
    max_percent = max(percentages)

    is_imbalanced = min_percent < 20 or max_percent > 60
    print(f"\nImbalance detected: {'Yes' if is_imbalanced else 'No'}")
    print(f"Min class: {min_percent:.1f}%, Max class: {max_percent:.1f}%")

    if is_imbalanced:
        print("💡 Recommendation: Use class_weight='balanced' in your model")
    else:
        print("✅ Classes are reasonably balanced")

    return distribution


def load_and_clean_data(csv_filepath):
    """
    Load workout data from CSV and perform initial cleaning.

    Args:
        csv_filepath (str): Path to the CSV file containing workout data

    Returns:
        pd.DataFrame: Cleaned DataFrame with standardized column names
    """
    print(f"📁 Loading data from: {csv_filepath}")

    # Load the raw data
    raw_dataframe = pd.read_csv(csv_filepath)
    print(f"   Loaded {len(raw_dataframe)} rows")

    # Standardize column names to expected format
    raw_dataframe.columns = ['prompt', 'completion']

    # Remove rows with missing values (ensures data quality)
    cleaned_dataframe = raw_dataframe.dropna()
    print(f"   After removing missing values: {len(cleaned_dataframe)} rows")

    return cleaned_dataframe


def extract_user_features_from_prompts(dataframe):
    """
    Extract user demographic and fitness characteristics from all prompt texts.

    Args:
        dataframe (pd.DataFrame): DataFrame containing 'prompt' column

    Returns:
        pd.DataFrame: DataFrame with added user characteristic columns
    """
    print("👤 Extracting user characteristics from prompts...")

    # Create a copy to avoid modifying original
    df_with_user_features = dataframe.copy()

    # Extract each user characteristic using regex patterns
    df_with_user_features['age'] = df_with_user_features['prompt'].str.extract(r'(\d+)-year-old').astype(int)
    df_with_user_features['gender'] = df_with_user_features['prompt'].str.extract(r'(male|female)')
    df_with_user_features['height'] = df_with_user_features['prompt'].str.extract(r'height of (\d+) cm').astype(int)
    df_with_user_features['weight'] = df_with_user_features['prompt'].str.extract(r'weight of (\d+) kg').astype(int)
    df_with_user_features['goal'] = df_with_user_features['prompt'].str.extract(r'goal is (.*?)(?:, and|\.)')
    df_with_user_features['gym_level'] = df_with_user_features['prompt'].str.extract(r'I am a (beginner|intermediate|advanced) at the gym')

    # Remove any rows where feature extraction failed
    df_with_user_features = df_with_user_features.dropna()
    print(f"   After feature extraction: {len(df_with_user_features)} rows")

    return df_with_user_features


def extract_workout_features_from_completions(dataframe):
    """
    Extract workout-specific features from all completion texts.

    Args:
        dataframe (pd.DataFrame): DataFrame containing 'completion' column

    Returns:
        pd.DataFrame: DataFrame with added workout feature columns
    """
    print("🏋️ Extracting workout features from completion texts...")

    # Extract features for each workout plan
    workout_features_list = []
    for completion_text in dataframe['completion']:
        workout_features = extract_workout_features(completion_text)
        workout_features_list.append(workout_features)

    # Convert to DataFrame and merge with original data
    workout_features_dataframe = pd.DataFrame(workout_features_list)
    enhanced_dataframe = pd.concat([dataframe, workout_features_dataframe], axis=1)

    print(f"   Added {len(workout_features_dataframe.columns)} workout features")

    return enhanced_dataframe


def create_target_variable(dataframe):
    """
    Create the target variable for classification from workout type.

    Args:
        dataframe (pd.DataFrame): DataFrame containing 'primary_workout_type' column

    Returns:
        tuple: (dataframe_with_target, label_encoder)
    """
    print("🎯 Creating target variable from workout types...")

    # Create a copy to avoid modifying original
    df_with_target = dataframe.copy()

    # Encode workout types as numerical labels
    target_label_encoder = LabelEncoder()
    df_with_target['workout_type'] = target_label_encoder.fit_transform(df_with_target['primary_workout_type'])

    # Display class mapping for transparency
    class_mapping = dict(zip(target_label_encoder.classes_, target_label_encoder.transform(target_label_encoder.classes_)))
    print(f"   Class mapping: {class_mapping}")

    # Analyze class distribution
    class_names = list(target_label_encoder.classes_)
    analyze_class_distribution(df_with_target['workout_type'], class_names)

    return df_with_target, target_label_encoder


def prepare_features_and_target(dataframe):
    """
    Prepare feature matrix (X) and target vector (y) for machine learning.

    Args:
        dataframe (pd.DataFrame): DataFrame with all features and target variable

    Returns:
        tuple: (X, y) where X is feature matrix and y is target vector
    """
    print("🔧 Preparing features and target for machine learning...")

    # Select features (exclude text columns and target variables)
    columns_to_exclude = ['prompt', 'completion', 'workout_type', 'primary_workout_type']
    feature_matrix = dataframe.drop(columns=columns_to_exclude, axis=1)
    target_vector = dataframe['workout_type']

    print(f"   Feature matrix shape: {feature_matrix.shape}")
    print(f"   Target vector shape: {target_vector.shape}")

    return feature_matrix, target_vector


def split_and_encode_data(X, y, test_size=0.2, random_state=42):
    """
    Split data and encode categorical features properly.

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        test_size (float): Proportion for test set
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test) with encoded features
    """
    print("✂️ Splitting data into train/test sets...")

    # Split the data first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")

    # Encode categorical features AFTER splitting to prevent data leakage
    print("🏷️ Encoding categorical features...")
    categorical_columns = ['gender', 'goal', 'gym_level', 'workout_duration', 'workout_intensity']

    X_train_encoded = pd.get_dummies(X_train, columns=categorical_columns)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_columns)

    # Ensure both datasets have the same columns (handle missing categories)
    train_columns = X_train_encoded.columns
    test_columns = X_test_encoded.columns

    # Add missing columns to training set
    missing_in_train = set(test_columns) - set(train_columns)
    for column in missing_in_train:
        X_train_encoded[column] = 0

    # Add missing columns to test set
    missing_in_test = set(train_columns) - set(test_columns)
    for column in missing_in_test:
        X_test_encoded[column] = 0

    # Ensure column order is consistent
    X_test_encoded = X_test_encoded[train_columns]

    print(f"   Final feature count: {len(train_columns)} features")

    return X_train_encoded, X_test_encoded, y_train, y_test


def normalize_numerical_features(X_train, X_test):
    """
    Normalize numerical features using MinMaxScaler.

    Args:
        X_train (pd.DataFrame): Training feature matrix
        X_test (pd.DataFrame): Test feature matrix

    Returns:
        tuple: (X_train_normalized, X_test_normalized)
    """
    print("📊 Normalizing numerical features...")

    # Define numerical features that need scaling
    numerical_feature_names = [
        'age', 'height', 'weight', 'total_exercises', 'total_volume',
        'rest_days', 'workout_days', 'avg_reps_per_exercise', 'exercise_variety'
    ]

    # Create copies to avoid modifying originals
    X_train_normalized = X_train.copy()
    X_test_normalized = X_test.copy()

    # Fit scaler on training data and transform both sets
    feature_scaler = MinMaxScaler()
    X_train_normalized[numerical_feature_names] = feature_scaler.fit_transform(X_train[numerical_feature_names])
    X_test_normalized[numerical_feature_names] = feature_scaler.transform(X_test[numerical_feature_names])

    print(f"   Normalized {len(numerical_feature_names)} numerical features")

    return X_train_normalized, X_test_normalized


def preprocess_data(csv_filepath, test_size=0.2, random_state=42):
    """
    Complete data preprocessing pipeline for workout recommendation.

    This function orchestrates the entire preprocessing workflow:
    1. Load and clean raw data
    2. Extract user characteristics from prompts
    3. Extract workout features from completions
    4. Create target variable
    5. Split data into train/test sets
    6. Encode categorical features
    7. Normalize numerical features

    Args:
        csv_filepath (str): Path to the CSV file containing workout data
        test_size (float): Proportion of data to use for testing (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)

    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Ready for machine learning

    Example:
        >>> X_train, X_test, y_train, y_test = preprocess_data('data/train.csv')
        >>> print(f"Ready for training: {X_train.shape}")
    """
    print("\n" + "="*60)
    print("🚀 STARTING WORKOUT DATA PREPROCESSING PIPELINE")
    print("="*60)

    # Step 1: Load and clean the raw data
    clean_dataframe = load_and_clean_data(csv_filepath)

    # Step 2: Extract user characteristics from prompt texts
    df_with_user_features = extract_user_features_from_prompts(clean_dataframe)

    # Step 3: Extract workout features from completion texts
    df_with_workout_features = extract_workout_features_from_completions(df_with_user_features)

    # Step 4: Create target variable for classification
    df_with_target, label_encoder = create_target_variable(df_with_workout_features)

    # Step 5: Prepare features and target for ML
    feature_matrix, target_vector = prepare_features_and_target(df_with_target)

    # Step 6: Split data and encode categorical features
    X_train_encoded, X_test_encoded, y_train, y_test = split_and_encode_data(
        feature_matrix, target_vector, test_size, random_state
    )

    # Step 7: Normalize numerical features
    X_train_final, X_test_final = normalize_numerical_features(X_train_encoded, X_test_encoded)

    print("\n" + "="*60)
    print("✅ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"📊 Final dataset shape: Train {X_train_final.shape}, Test {X_test_final.shape}")
    print(f"🎯 Target classes: {len(set(y_train))} unique workout types")

    return X_train_final, X_test_final, y_train, y_test


if __name__ == '__main__':
    # Example usage - test with smaller sample first
    print("Testing data preprocessing pipeline...")

    # Load a small sample to test
    import pandas as pd
    df_full = pd.read_csv('../data/train.csv')
    print(f"Full dataset size: {len(df_full)} rows")

    # Test with first 100 rows
    df_sample = df_full.head(100)
    df_sample.to_csv('../data/train_sample.csv', index=False)

    print("Testing with sample data (100 rows)...")
    X_train, X_test, y_train, y_test = preprocess_data('../data/train_sample.csv')
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("Feature columns:", list(X_train.columns))
    print("Target classes:", sorted(set(y_train)))
