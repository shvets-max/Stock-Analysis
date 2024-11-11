import os
import pandas as pd

from constants import METRICS_DIR

METRICS_WEIGHTS = {
    "Operating Margin": 5,
    "Total Debt": 3,
    "EBITDA": 8,
    "CFO": 5,
    "EV": 2,
    "Stock Price": 0,

    # derivatives
    "Total Debt / EBITDA": 7,
    "Total Debt / CFO": 7,
    "Price / Sales": 10,
    "Price / CFO": 9,
    "Price / EV": 8,
}

# ----------
# Directories
WORKDIR = os.path.join(".")
DYNAMIC_METRICS_DIR = os.path.join(METRICS_DIR, "dynamic_metrics")


def prepare_time_series(df: pd.DataFrame):
    df.index = pd.to_datetime(df.index)
    df = df[df.index.year >= 2014]
    df = df.map(lambda x: str(x).replace(",", ".")).astype(float)
    df.fillna(method="ffill", inplace=True)
    df.fillna(0, inplace=True)
    return df


_margin = pd.read_csv(os.path.join(DYNAMIC_METRICS_DIR, "operating-margin.csv"), index_col=0, sep=";", on_bad_lines='warn')
margin = prepare_time_series(_margin)

_debt = pd.read_csv(os.path.join(DYNAMIC_METRICS_DIR, "total-debt.csv"), index_col=0, sep=";", on_bad_lines='warn')
debt = prepare_time_series(_debt)

_ebitda = pd.read_csv(os.path.join(DYNAMIC_METRICS_DIR, "ebitda.csv"), index_col=0, sep=";", on_bad_lines='warn')
ebitda = prepare_time_series(_ebitda)

_cfo = pd.read_csv(os.path.join(DYNAMIC_METRICS_DIR, "cfo.csv"), index_col=0, sep=";", on_bad_lines='warn')
cfo = prepare_time_series(_cfo)

debt_ebitda = debt / ebitda

