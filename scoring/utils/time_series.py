import pandas as pd
import csv
import datetime


def read_csv_to_df(filepath: str) -> pd.DataFrame:
    with open(filepath) as f:
        dialect = csv.Sniffer().sniff(f.readline(), delimiters=",;")

    df = pd.read_csv(filepath, index_col=0, sep=dialect.delimiter, on_bad_lines='warn')
    return df

def prepare_time_series(df: pd.DataFrame):
    df.index = pd.to_datetime(df.index)
    df = df[df.index.year >= 2015]

    # Fill date gaps
    full_date_range = pd.date_range(start=df.index.min(), end=datetime.date.today())
    df = df.reindex(full_date_range)
    df = df.ffill()

    df = df.map(lambda x: str(x).replace(",", ".")).astype(float)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    return df