import os

import numpy as np
import pandas as pd

from constants import METRICS_DIR
from scoring.utils.time_series import read_csv_to_df, prepare_time_series

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
DATA_SOURCES = [
    "revenue.csv", "stock-price.csv", "shares-outstanding.csv",
    "operating-margin.csv", "total-debt.csv", "ebitda.csv", "cfo.csv", "total-cash.csv"
]

data = {
    source.rstrip(".csv"): prepare_time_series(read_csv_to_df(os.path.join(DYNAMIC_METRICS_DIR, source)))
    for source in DATA_SOURCES
}

data["marcap"] = data["stock-price"] * data["shares-outstanding"]
data["ev"] = data["marcap"] + data["total-debt"] - data["total-cash"]
data["debt_to_ebitda"] = data["total-debt"] / data["ebitda"]
data["debt_to_cfo"] = data["total-debt"] / data["cfo"]
data["marcap_to_sales"] = data["marcap"] / data["revenue"]
data["marcap_to_cfo"] = data["marcap"] / data["cfo"]
data["ev_to_ebitda"] = data["ev"] / data["ebitda"]

def calculate_mean_minus_std(
        series: pd.DataFrame,
        is_percentages: bool = False,
        last_quarters: int = 12,
        qoq: bool = False
):
    """
    Calculate mean growth rate subtracted by std.
    This metric tells how high is the growth rate and how regular it is.
    :param series:
    :param is_percentages: (bool) is already series representing percentages.
    :param last_quarters:
    :param qoq: (bool) calculate quarter-over-quarter.
    :return:
    """
    unique_vals = series[np.where(series.diff() != 0)[0]]
    if is_percentages:
        perc_chg = unique_vals.diff(periods=4)[-last_quarters:] if qoq else unique_vals.diff()[-last_quarters:]
    else:
        perc_chg = unique_vals.pct_change(periods=4)[-last_quarters:] if qoq else unique_vals.pct_change()[-last_quarters:]

    mean_chg = perc_chg.mean()
    return mean_chg - perc_chg.std() / 2

def normalize_ewm_growth(series: pd.Series, timerange: int = 365, low: float = .0, high: float = .05) -> pd.Series:
    """
    Calculate exponential moving average and normalize it by a given value range.
    :param timerange:
    :param series:
    :param low:
    :param high:
    :return:
    """

    unique_vals = series[-timerange:][np.where(series[-timerange:].diff() != 0)[0]]
    perc_chg = unique_vals.pct_change()
    ewm = perc_chg.ewm(alpha=0.5).mean()
    norm_ewm = (ewm - low) / (high - low)
    return norm_ewm

def normalize_growth_rate(series: pd.Series, timerange: int = 365, low: float = .0, high: float = .05) -> pd.Series:
    """
    Calculate exponential moving average and normalize it by a given value range.
    :param timerange:
    :param series:
    :param low:
    :param high:
    :return:
    """

    unique_vals = series[-timerange:][np.where(series[-timerange:].diff() != 0)[0]]
    perc_chg = unique_vals.pct_change()
    norm_chg = (perc_chg - low) / (high - low)
    return norm_chg

def normalize_by_quantile_band(
        series: pd.Series,
        timerange: int = 365,
        q_low: float = .2,
        q_high: float = .8
) -> float:
    """
    Normalize last datapoint to quantile band along the given timerange.

    :param series:
    :param timerange:
    :param q_low:
    :param q_high:
    :return:
    """
    qh = series[-timerange:].quantile(q_high)
    ql = series[-timerange:].quantile(q_low)
    return (series[-1] - ql) / (qh - ql)


def normalize_by_mean(
        series: pd.Series,
        timerange: int = 365
) -> float:
    """
    Normalize last datapoint to quantile band along the given timerange.

    :param series:
    :param timerange:
    :return:
    """
    mean = series[-timerange:].mean()
    return (series[-1] - mean) / mean

def calculate_debt_dynamic_score(series: pd.Series, threshold: float, smoothing: bool = True, n_quarters: int = 5) -> float:
    """
    Measure debt dynamic. 0 - good, 1 - bad.
    :param n_quarters:
    :param smoothing:
    :param threshold:
    :param series:
    :return:
    """
    # 1. If above threshold
    above_thresh = series[-1] > threshold

    # 2. Is last 5 quarters debt growth is "under control"
    if smoothing:
        series_norm = normalize_ewm_growth(series, 3*365, low=0.02, high=0.1)
    else:
        series_norm = normalize_growth_rate(series, 3*365, low=0.02, high=0.1)

    # Averages over last 1, 2, ... , n-quarters.
    qs = np.array([series_norm[-q:].mean() for q in range(n_quarters, 0, -1)])
    # Debt-under-control quarters (last 1...n-quarters averages under 0.1)
    duc_quarters = (qs <= 1).mean()

    # 3. Non-increasing debt score. 0 - increasing all the time, 1 - non-increasing all the time.
    epsilon = 0.1
    debt_down_points = 0
    for x in range(n_quarters - 1):
        debt_down_points += (qs[x] + epsilon >= qs[x+1:]).sum()
    debt_down_ratio = debt_down_points / 10

    # 1st component tells if debt is high,
    # 2nd and 3rd components tell whether company keeps it under control and reduces it regularly.
    return float(above_thresh) * (1 - (duc_quarters + debt_down_ratio) / 2)


def calculate_scores(ticker: str):
    out = {}

    # Growth metrics
    # Objective: maximize mean - std (to reward companies, for which a metric is not decreasing (mean chg.) and growing stably.
    ebitda_mean_minus_std = calculate_mean_minus_std(data["ebitda"][ticker], last_quarters=8)
    ebitda_qoq_mean_minus_std = calculate_mean_minus_std(data["ebitda"][ticker], last_quarters=8, qoq=True)

    revenue_mean_minus_std = calculate_mean_minus_std(data["revenue"][ticker], last_quarters=8)
    revenue_qoq_mean_minus_std = calculate_mean_minus_std(data["revenue"][ticker], last_quarters=8, qoq=True)

    cfo_mean_minus_std = calculate_mean_minus_std(data["cfo"][ticker], last_quarters=8)
    cfo_qoq_mean_minus_std = calculate_mean_minus_std(data["cfo"][ticker], last_quarters=8, qoq=True)

    op_margin_mean_minus_std = calculate_mean_minus_std(data["operating-margin"][ticker], is_percentages=True, last_quarters=8)
    op_margin_qoq_mean_minus_std = calculate_mean_minus_std(data["operating-margin"][ticker], is_percentages=True, last_quarters=8, qoq=True)

    # Debt metrics
    # Objective: to punish only companies with high debt,
    # but reward if they don't increase it rapidly and/or regularly reduce it.
    debt_to_cfo_score = calculate_debt_dynamic_score(data["debt_to_cfo"][ticker], threshold=3)
    debt_to_ebitda_score = calculate_debt_dynamic_score(data["debt_to_ebitda"][ticker], threshold=3)

    # Valuation metrics:
    # Objective: reward underpriced, punish overpriced.
    ps_score = normalize_by_quantile_band(data["marcap_to_sales"][ticker], 3 * 365, q_low=0.05, q_high=0.95)
    p_cfo_score = normalize_by_quantile_band(data["marcap_to_cfo"][ticker], 3 * 365, q_low=0.05, q_high=0.95)
    ev_ebitda_score = normalize_by_quantile_band(data["ev_to_ebitda"][ticker], 3 * 365, q_low=0.05, q_high=0.95)

    # Growth metrics: revenue, CFO, EBITDA and Operating Margin
    out["ebitda_growth_score"] = np.clip(ebitda_mean_minus_std, 0, .15)
    out["ebitda_qoq_growth_score"] = np.clip(ebitda_qoq_mean_minus_std, 0, .15)

    out["revenue_growth_score"] = np.clip(revenue_mean_minus_std, 0, .15)
    out["revenue_qoq_growth_score"] = np.clip(revenue_qoq_mean_minus_std, 0, .15)

    out["cfo_growth_score"] = np.clip(cfo_mean_minus_std, 0, .15)
    out["cfo_qoq_growth_score"] = np.clip(cfo_qoq_mean_minus_std, 0, .15)

    out["op_margin_growth_score"] = np.clip(op_margin_mean_minus_std, -.02, .05)
    out["op_margin_qoq_growth_score"] = np.clip(op_margin_qoq_mean_minus_std, -.02, .05)

    # Debt metrics
    out["debt_to_cfo_score"] = 1 - debt_to_cfo_score
    out["debt_to_ebitda_score"] = 1 - debt_to_ebitda_score

    # Valuation metrics
    out["ps_score"] = np.clip(1 - ps_score, 0, 1)
    out["p_cfo_score"] = np.clip(1 - p_cfo_score, 0, 1)
    out["ev_ebitda_score"] = np.clip(1 - ev_ebitda_score, 0, 1)

    return out

tickers_scores = {
    ticker: calculate_scores(ticker)
    for ticker in data["revenue"].columns
}

df_scores = pd.DataFrame.from_dict(tickers_scores)

# Plots
# import matplotlib.pyplot as plt
# data["marcap_to_sales"].plot.line()
# plt.savefig("marcap_to_sales.png")

