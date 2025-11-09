import pandas as pd
import os

def load_data(path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is not a valid CSV or missing columns.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Could not read file: {e}")

    expected_cols = {"text", "label"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {expected_cols - set(df.columns)}")

    return df
