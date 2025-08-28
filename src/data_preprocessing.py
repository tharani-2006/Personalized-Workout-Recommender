import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


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

    # Encode target variable (workout_type) - Assuming workout_type can be derived from completion, for now just use a dummy target
    df['workout_type'] = 0  # Replace with actual target derivation logic
    label_encoder = LabelEncoder()
    df['workout_type'] = label_encoder.fit_transform(df['workout_type'])

    # Prepare data for splitting
    X = df.drop(['prompt', 'completion', 'workout_type'], axis=1)
    y = df['workout_type']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Encode categorical features (Do this *after* the train/test split)
    X_train = pd.get_dummies(X_train, columns=['gender', 'goal', 'gym_level'])
    X_test = pd.get_dummies(X_test, columns=['gender', 'goal', 'gym_level'])

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
    numerical_features = ['age', 'height', 'weight']
    scaler = MinMaxScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features]) # Fit AND transform train
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])       # Only transform test

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Example usage
    X_train, X_test, y_train, y_test = preprocess_data('data/train.csv')
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
