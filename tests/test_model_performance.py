from src.load_data import load_heart_disease
from src.preprocess import preprocess_data
from src.train_model import train_model

def test_model_performance():
    """
    Test to verify model performance meets baseline threshold.
    """

    # Load dataset
    df = load_heart_disease()

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train model and get accuracy
    model, accuracy = train_model(X_train, X_test, y_train, y_test)

    # Baseline performance threshold
    baseline_accuracy = 0.70

    assert accuracy >= baseline_accuracy, (
        f"Model accuracy {accuracy:.2f} is below baseline {baseline_accuracy}"
    )


