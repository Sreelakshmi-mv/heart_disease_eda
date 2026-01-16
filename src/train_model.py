import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.preprocess import preprocess_data

# Configure logging
logging.basicConfig(level=logging.INFO)

def train_and_evaluate():
    """
    Trains a baseline Logistic Regression model
    and returns accuracy score.
    """
    logging.info("Starting model training")

    X, y = preprocess_data()

    # Train-test split (reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Baseline model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logging.info(f"Model training completed. Accuracy: {accuracy}")

    return accuracy


if __name__ == "__main__":
    acc = train_and_evaluate()
    print("Model Accuracy:", acc)
