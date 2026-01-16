import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_heart_disease():
    """
    Loads the Heart Disease dataset from CSV file.
    Returns a pandas DataFrame.
    """
    try:
        data_path = os.path.join("data", "heart.csv")

        logging.info("Loading heart disease dataset from CSV")
        df = pd.read_csv(data_path)

        logging.info("Dataset loaded successfully")
        return df

    except Exception as e:
        logging.error("Failed to load dataset", exc_info=True)
        raise


if __name__ == "__main__":
    df = load_heart_disease()
    print(df.head())
    print("Dataset shape:", df.shape)
