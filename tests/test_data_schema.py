from src.load_data import load_heart_disease

def test_dataset_schema():
    """
    Test to validate dataset integrity and schema.
    """
    df = load_heart_disease()

    # Dataset should not be empty
    assert not df.empty, "Dataset is empty"

    # Expected columns in Heart Disease dataset
    expected_columns = {
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "num"
    }

    # Check if expected columns exist
    assert expected_columns.issubset(set(df.columns)), \
        "Dataset schema does not match expected structure"
