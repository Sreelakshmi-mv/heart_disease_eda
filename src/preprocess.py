import logging
from src.load_data import load_heart_disease
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)

def preprocess_data():
    """
    Preprocesses the heart disease dataset:
    - Removes missing values
    - Separates features and target
    - Scales numerical features
    Returns processed DataFrame.
    """
    logging.info("Starting preprocessing")

    df = load_heart_disease()

    # Drop missing values
    df = df.dropna()

    # Separate target
    X = df.drop("num", axis=1)
    y = df["num"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logging.info("Preprocessing completed")

    return X_scaled, y


if __name__ == "__main__":
    X, y = preprocess_data()
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
