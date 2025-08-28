import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support
from data_preprocessing import preprocess_data

def train_model(X_train, y_train):
    """Trains a Random Forest model."""
    model = RandomForestClassifier(random_state=42)  # You can tune hyperparameters here
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


if __name__ == '__main__':
    # Load and preprocess data (replace with your actual data loading)
    X_train, X_test, y_train, y_test = preprocess_data('data/train.csv')

    # Train the model
    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

