import pandas as pd


def load_and_normalize_percentages(filepath: str) -> pd.DataFrame:
    """
    Loads a CSV file into a Pandas DataFrame, normalizes specified percentage columns,
    and returns the DataFrame.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The DataFrame with normalized percentage columns.
                      Returns the original DataFrame if specified columns are not found.
    """
    df = pd.read_csv(filepath, index_col=0)
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                # Try to convert the column to numeric, handling percentages
                df[col] = (
                    df[col].astype(str).str.replace("%", "", regex=False).astype(float)
                    / 100
                )
            except ValueError:
                # If conversion fails, skip the column
                print(
                    f"Warning: Could not convert column '{col}' to numeric."
                    f" Skipping normalization."
                )
                continue
        elif pd.api.types.is_numeric_dtype(df[col]):
            # If the column is already numeric, check if it's in percentage format
            # (values between 0 and 100)
            if df[col].min() >= 0 and df[col].max() <= 100:
                df[col] = df[col] / 100
    return df
