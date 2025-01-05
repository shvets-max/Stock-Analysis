import datetime
import os
import csv

import numpy as np
import pandas as pd

from constants import METRICS_DIR

METRICS_WEIGHTS = {
    "Total Debt": 0,
    "Total Cash": 0,
    "CFO": 0,
    "Revenue": 0,
    "Marcap": 0,

    "EBITDA": 2,
    "Operating Margin": 5,

    # derivatives
    "EV": 0,  # EV = Marcap + Total Debt - Total Cash

    "Total Debt / EBITDA": 7,
    "Total Debt / CFO": 7,
    "Marcap / Sales": 10,  # Marcap / Revenue
    "Marcap / CFO": 9, # Marcap / CFO
    "EV / EBITDA": 8,
}

# ----------
# Directories
WORKDIR = os.path.join(".")
DYNAMIC_METRICS_DIR = os.path.join(METRICS_DIR, "dynamic_metrics")
DATA_SOURCES = ["revenue.csv", "marcap.csv", "operating-margin.csv", "total-debt.csv", "ebitda.csv", "cfo.csv", "total-cash.csv"]

def read_csv_to_df(filename: str) -> pd.DataFrame:
    with open(os.path.join(DYNAMIC_METRICS_DIR, filename)) as f:
        dialect = csv.Sniffer().sniff(f.readline(), delimiters=",;")

    df = pd.read_csv(os.path.join(DYNAMIC_METRICS_DIR, filename), index_col=0, sep=dialect.delimiter, on_bad_lines='warn')
    return df

def prepare_time_series(df: pd.DataFrame):
    df.index = pd.to_datetime(df.index)
    df = df[df.index.year >= 2015]

    # Fill date gaps
    full_date_range = pd.date_range(start=df.index.min(), end=datetime.date.today())
    df = df.reindex(full_date_range)
    df = df.ffill()

    df = df.map(lambda x: str(x).replace(",", ".")).astype(float)
    df.fillna(method="ffill", inplace=True)
    df.fillna(0, inplace=True)
    return df

data = {
    source.rstrip(".csv"): prepare_time_series(read_csv_to_df(source))
    for source in DATA_SOURCES
}

data["ev"] = data["marcap"] + data["total-debt"] - data["total-cash"]
data["debt_to_ebitda"] = data["total-debt"] / data["ebitda"]
data["debt_to_cfo"] = data["total-debt"] / data["cfo"]
data["marcap_to_sales"] = data["marcap"] / data["revenue"]
data["marcap_to_cfo"] = data["marcap"] / data["cfo"]
data["ev_to_ebitda"] = data["ev"] / data["ebitda"]

# EBITDA and Operating Margin
# Objective: not decreasing (avg. chg.), stable growth (std)
def calculate_avg_std(series: pd.DataFrame, timestamp: int = 365):
    unique_vals = series[-timestamp:][np.where(series[-timestamp:].diff() != 0)[0]]
    perc_chg = (unique_vals - unique_vals.shift()) / unique_vals
    avg_chg = perc_chg[-6:].mean()
    std_to_mean = perc_chg.std() / avg_chg

    return avg_chg, std_to_mean

ebitda_chg, ebitda_std = calculate_avg_std(data["ebitda"]["$MRK"], 3*365)
op_margin_chg, op_margin_std = calculate_avg_std(data["operating-margin"]["$MRK"], 3*365)

# Debt metrics
# (desirable: not a high debt or not increasing rapidly and regularly decreased)
def normalize_ewm_growth(series: pd.Series, timerange: int = 365, low: float = .0, high: float = .05) -> pd.Series:
    """

    :param timerange:
    :param series:
    :param low:
    :param high:
    :return:
    """

    unique_vals = series[-timerange:][np.where(series[-timerange:].diff() != 0)[0]]
    perc_chg = (unique_vals - unique_vals.shift()) / unique_vals
    ewm = perc_chg.ewm(alpha=0.5).mean()
    norm_ewm = (ewm - low) / (high - low)
    return norm_ewm

def calculate_debt_dynamic_score(series: pd.Series, threshold: float) -> float:
    """
    Measure debt dynamic. 0 - good, 1 - bad.
    :param threshold:
    :param series:
    :return:
    """
    # 1. If above threshold
    above_thresh = series[-1] > threshold

    # 2. Is last 5 quarters debt growth is "under control" (for how many quarters 3Y ewm < threshold)
    series_norm_ewm = normalize_ewm_growth(series, 3*365, low=-0.02, high=0.05)
    q5 = np.array([series_norm_ewm[-q:].mean() for q in range(5, 0, -1)])
    debt_under_control = (q5 <= 1).mean()

    # 3. Non-increasing debt score. 0 - increasing all the time, 1 - non-increasing all the time.
    epsilon = 0.1
    decrease_points = 0
    for x in range(4):
        decrease_points += (q5[x] + epsilon >= q5[x+1:]).sum()
    decrease_ratio = decrease_points / 10

    # 1st component tells if debt is high,
    # 2nd and 3rd components tell whether company keeps it under control and reduces it regularly.
    return float(above_thresh) * (1 - (debt_under_control + decrease_ratio) / 2)

debt_to_cfo_score = calculate_debt_dynamic_score(data["debt_to_cfo"]["$MRK"], threshold=3)
debt_to_ebitda_score = calculate_debt_dynamic_score(data["debt_to_ebitda"]["$MRK"], threshold=3)

# Valuation metrics:
def normalize_by_rolling_quantile_band(
        series: pd.Series,
        timerange: int = 365,
        q_low: float = .2,
        q_high: float = .8
) -> float:
    """
    Normalize last datapoint to quantile band along given timerange.

    :param series:
    :param timerange:
    :param q_low:
    :param q_high:
    :return:
    """
    qh = series[-timerange:].quantile(q_high)
    ql = series[-timerange:].quantile(q_low)
    return (series[-1] - ql) / (qh - ql)

ps_score = normalize_by_rolling_quantile_band(data["marcap_to_sales"]["$MRK"], 3*365)
p_cfo_score = normalize_by_rolling_quantile_band(data["marcap_to_cfo"]["$MRK"], 3*365)
ev_ebitda_score = normalize_by_rolling_quantile_band(data["ev_to_ebitda"]["$MRK"], 3*365)

# Plots
# import matplotlib.pyplot as plt
# data["marcap_to_sales"].plot.line()
# plt.savefig("marcap_to_sales.png")

