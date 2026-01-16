from src.load_data import load_heart_disease
from src.preprocess import preprocess_data
from src.train_model import train_model

def test_pipeline_reproducibility():
    """
    Test to verify that same inputs produce same outputs.
    """

    # Load dataset
    df = load_heart_disease()

    # First run
    X_train1, X_test1, y_train1, y_test1 = preprocess_data(df)
    model1, acc1 = train_model(X_train1, X_test1, y_train1, y_test1)

    # Second run (same data, same random state)
    X_train2, X_test2, y_train2, y_test2 = preprocess_data(df)
    model2, acc2 = train_model(X_train2, X_test2, y_train2, y_test2)

    # Accuracy should be the same
    assert acc1 == acc2, "Pipeline is not reproducible"
