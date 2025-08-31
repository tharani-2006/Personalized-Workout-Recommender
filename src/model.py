import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from data_preprocessing import preprocess_data

def tune_hyperparameters(X_train, y_train, search_type='grid', n_iter=50, use_class_weight=True):
    """
    Tunes the Random Forest hyperparameters using GridSearchCV or RandomizedSearchCV.

    Args:
        X_train: Training features
        y_train: Training labels
        search_type: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV
        n_iter: Number of iterations for RandomizedSearchCV
        use_class_weight: Whether to use balanced class weights

    Returns:
        Best estimator from the search
    """
    # Check class distribution to determine if we need class balancing
    unique, counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    percentages = [(count / total_samples) * 100 for count in counts]
    min_percent = min(percentages)
    max_percent = max(percentages)
    is_imbalanced = min_percent < 20 or max_percent > 60

    print(f"Class distribution: {dict(zip(unique, counts))}")
    print(f"Class imbalance detected: {'Yes' if is_imbalanced else 'No'}")

    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 4, 8],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    # Create base model with class weight handling
    class_weight = 'balanced' if (use_class_weight and is_imbalanced) else None
    if class_weight:
        print("🎯 Using balanced class weights to handle imbalanced data")
    else:
        print("✅ Using default class weights (data appears balanced)")

    rf = RandomForestClassifier(random_state=42, class_weight=class_weight)

    if search_type == 'grid':
        print("Performing Grid Search...")
        search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,  # 3-fold cross-validation
            scoring='accuracy',
            n_jobs=-1,  # Use all available cores
            verbose=1
        )
    else:  # random search
        print("Performing Randomized Search...")
        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )

    # Fit the search
    search.fit(X_train, y_train)

    print("Best Hyperparameters:", search.best_params_)
    print("Best Cross-Validation Score:", search.best_score_)

    return search.best_estimator_


def train_model(X_train, y_train, tune_hyperparams=False, search_type='random', use_class_weight=True):
    """
    Trains a Random Forest model.

    Args:
        X_train: Training features
        y_train: Training labels
        tune_hyperparams: Whether to tune hyperparameters
        search_type: 'grid' or 'random' for hyperparameter search
        use_class_weight: Whether to use balanced class weights

    Returns:
        Trained model
    """
    if tune_hyperparams:
        print("Training model with hyperparameter tuning...")
        model = tune_hyperparameters(X_train, y_train, search_type=search_type, use_class_weight=use_class_weight)
    else:
        print("Training model with default hyperparameters...")

        # Check if we need class balancing for baseline model too
        unique, counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        percentages = [(count / total_samples) * 100 for count in counts]
        min_percent = min(percentages)
        max_percent = max(percentages)
        is_imbalanced = min_percent < 20 or max_percent > 60

        class_weight = 'balanced' if (use_class_weight and is_imbalanced) else None
        if class_weight:
            print("🎯 Using balanced class weights for baseline model")

        model = RandomForestClassifier(random_state=42, class_weight=class_weight)
        model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

    # Calculate precision, recall, fscore, and support
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {fscore}")


def save_model_and_preprocessor(model, model_path='../models/workout_model.pkl'):
    """
    Save the trained model for later use in API.

    Args:
        model: Trained machine learning model
        model_path: Path where to save the model
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"💾 Model saved to: {model_path}")


def train_and_save_production_model(csv_filepath='../data/train.csv'):
    """
    Train the final production model and save it for API use.

    Args:
        csv_filepath: Path to training data

    Returns:
        tuple: (trained_model, X_test, y_test) for final evaluation
    """
    print("\n" + "="*60)
    print("🏭 TRAINING PRODUCTION MODEL FOR API DEPLOYMENT")
    print("="*60)

    # Load and preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(csv_filepath)

    # Train model with hyperparameter tuning and class balancing
    production_model = train_model(
        X_train, y_train,
        tune_hyperparams=True,
        search_type='random',
        use_class_weight=True
    )

    # Evaluate the production model
    print("\n" + "="*50)
    print("📊 PRODUCTION MODEL EVALUATION")
    print("="*50)
    evaluate_model(production_model, X_test, y_test)

    # Save the model for API use
    save_model_and_preprocessor(production_model)

    return production_model, X_test, y_test


if __name__ == '__main__':
    # Train and save the production model for API deployment
    production_model, X_test, y_test = train_and_save_production_model()

    print("\n" + "🎉" + "="*58 + "🎉")
    print("🚀 PRODUCTION MODEL READY FOR API DEPLOYMENT!")
    print("🎉" + "="*58 + "🎉")
    print("📁 Model saved in: models/workout_model.pkl")
    print("🌐 Ready to create Flask API endpoint!")
    print("💡 Next step: Create app.py for web API")

