import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def extract_workout_features(completion_text):
    """
    Extract workout duration and intensity features from completion text.

    Args:
        completion_text (str): The workout plan text

    Returns:
        dict: Dictionary containing extracted features
    """
    features = {}

    # Count total number of exercises (excluding rest days)
    exercise_pattern = r'- ([^:]+): (\d+) sets of (\d+) reps'
    exercises = re.findall(exercise_pattern, completion_text)
    features['total_exercises'] = len(exercises)

    # Calculate total volume (sets * reps)
    total_volume = sum(int(sets) * int(reps) for _, sets, reps in exercises)
    features['total_volume'] = total_volume

    # Count rest days
    rest_days = len(re.findall(r'Rest Day', completion_text))
    features['rest_days'] = rest_days

    # Calculate workout days (7 - rest days)
    features['workout_days'] = 7 - rest_days

    # Determine workout duration category based on total volume
    if total_volume <= 200:
        features['workout_duration'] = 'short'
    elif total_volume <= 400:
        features['workout_duration'] = 'medium'
    else:
        features['workout_duration'] = 'long'

    # Calculate average reps per exercise (intensity indicator)
    if exercises:
        avg_reps = np.mean([int(reps) for _, _, reps in exercises])
        features['avg_reps_per_exercise'] = avg_reps

        # Determine intensity based on average reps
        if avg_reps <= 8:
            features['workout_intensity'] = 'high'  # Low reps = high intensity
        elif avg_reps <= 15:
            features['workout_intensity'] = 'medium'
        else:
            features['workout_intensity'] = 'low'   # High reps = low intensity
    else:
        features['avg_reps_per_exercise'] = 0
        features['workout_intensity'] = 'low'

    # Count different exercise types for variety
    exercise_names = [exercise[0].strip() for exercise in exercises]
    features['exercise_variety'] = len(set(exercise_names))

    # Identify workout type based on exercises
    cardio_exercises = ['Running', 'Cycling', 'Walking', 'Jump Rope', 'Jumping Jacks',
                       'High Knees', 'Mountain Climbers', 'Burpees']
    strength_exercises = ['Bench Press', 'Deadlifts', 'Squats', 'Pull-ups', 'Push-ups',
                         'Dumbbell Rows', 'Shoulder Press', 'Bicep Curls', 'Tricep Dips']

    cardio_count = sum(1 for name in exercise_names if any(cardio in name for cardio in cardio_exercises))
    strength_count = sum(1 for name in exercise_names if any(strength in name for strength in strength_exercises))

    if cardio_count > strength_count:
        features['primary_workout_type'] = 'cardio'
    elif strength_count > cardio_count:
        features['primary_workout_type'] = 'strength'
    else:
        features['primary_workout_type'] = 'mixed'

    return features


def preprocess_data(csv_filepath, test_size=0.2, random_state=42):
    """
    Preprocesses the data from the given CSV file.

    Args:
        csv_filepath (str): The path to the CSV file.
        test_size (float): The proportion of the data to use for the test set.
        random_state (int): The random state to use for the train/test split.

    Returns:
        tuple: A tuple containing the training and testing sets (X_train, X_test, y_train, y_test).
    """

    # Load the data
    df = pd.read_csv(csv_filepath)

    # Rename columns to 'prompt' and 'completion'
    df.columns = ['prompt', 'completion']

    # Handle missing values (drop rows with missing values for simplicity)
    df = df.dropna()

    # Feature Engineering (extract features from the prompt column)
    df['age'] = df['prompt'].str.extract(r'(\d+)-year-old').astype(int)
    df['gender'] = df['prompt'].str.extract(r'(male|female)')
    df['height'] = df['prompt'].str.extract(r'height of (\d+) cm').astype(int)
    df['weight'] = df['prompt'].str.extract(r'weight of (\d+) kg').astype(int)
    df['goal'] = df['prompt'].str.extract(r'goal is (.*?)(?:, and|\.)')
    df['gym_level'] = df['prompt'].str.extract(r'I am a (beginner|intermediate|advanced) at the gym')

    # Handle missing values after feature engineering
    df = df.dropna()

    # Extract workout features from completion text
    print("Extracting workout features from completion text...")
    workout_features_list = []
    for completion in df['completion']:
        workout_features = extract_workout_features(completion)
        workout_features_list.append(workout_features)

    # Convert to DataFrame and merge
    workout_features_df = pd.DataFrame(workout_features_list)
    df = pd.concat([df, workout_features_df], axis=1)

    # Create target variable from primary_workout_type (this is our classification target)
    label_encoder = LabelEncoder()
    df['workout_type'] = label_encoder.fit_transform(df['primary_workout_type'])

    # Prepare data for splitting (drop text columns and target, keep primary_workout_type for encoding)
    X = df.drop(['prompt', 'completion', 'workout_type', 'primary_workout_type'], axis=1)
    y = df['workout_type']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Encode categorical features (Do this *after* the train/test split)
    categorical_columns = ['gender', 'goal', 'gym_level', 'workout_duration', 'workout_intensity']
    X_train = pd.get_dummies(X_train, columns=categorical_columns)
    X_test = pd.get_dummies(X_test, columns=categorical_columns)

    # Ensure both dataframes have the same columns
    train_cols = X_train.columns
    test_cols = X_test.columns

    missing_cols_train = set(test_cols) - set(train_cols)
    for c in missing_cols_train:
        X_train[c] = 0

    missing_cols_test = set(train_cols) - set(test_cols)
    for c in missing_cols_test:
        X_test[c] = 0

    # Ensure the order of column is the same
    X_test = X_test[train_cols]

    # Normalize numerical features (Do this *after* the train/test split)
    numerical_features = ['age', 'height', 'weight', 'total_exercises', 'total_volume',
                         'rest_days', 'workout_days', 'avg_reps_per_exercise', 'exercise_variety']
    scaler = MinMaxScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features]) # Fit AND transform train
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])       # Only transform test

    return X_train, X_test, y_train, y_test


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
