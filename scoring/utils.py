import pandas as pd
from typing import List, Tuple


def load_and_normalize_percentages(filepath: str, norm_columns: List| Tuple) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0)
    df[norm_columns] = df[norm_columns].map(
        lambda x: str(x).replace("%", "")
    ).astype(float) / 100

    return df
