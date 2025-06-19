import logging
import os

import numpy as np
import pandas as pd
from constants import METRICS_DIR

from scoring.utils.time_series import prepare_time_series, read_csv_to_df

# Define constants for better readability
DEFAULT_TIMERANGE = 365
Q_LOW = 0.05
Q_HIGH = 0.95
DEBT_THRESHOLD = 3
EWM_LOW = 0.02
EWM_HIGH = 0.1
N_QUARTERS = 5
EBITDA_CLIP = 0.15
OP_MARGIN_CLIP_LOW = -0.02
OP_MARGIN_CLIP_HIGH = 0.05
DEFAULT_ALPHA = 0.5  # Alpha for exponential moving average

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
    "Marcap / CFO": 9,  # Marcap / CFO
    "EV / EBITDA": 8,
}

# ----------
# Directories
WORKDIR = os.path.join(".")

# Define data sources
DATA_SOURCES = [
    "revenue.csv",
    "stock-price.csv",
    "shares-outstanding.csv",
    "operating-margin.csv",
    "total-debt.csv",
    "ebitda.csv",
    "cfo.csv",
    "total-cash.csv",
]

# Define directories
DYNAMIC_METRICS_DIR = os.path.join(METRICS_DIR, "dynamic_metrics")

data = {
    source.rstrip(".csv"): prepare_time_series(
        read_csv_to_df(os.path.join(DYNAMIC_METRICS_DIR, source))
    )
    for source in DATA_SOURCES
}


def load_and_prepare_data(data_dir: str, data_sources: list) -> dict:
    """
    Loads data from CSV files, prepares time series, and calculates derived metrics.

    Args:
        data_dir (str): The directory containing the CSV files.
        data_sources (list): A list of CSV filenames.

    Returns:
        dict: A dictionary containing the prepared data.
    """
    data = {}
    for source in data_sources:
        metric_name = source.rstrip(".csv")
        file_path = os.path.join(data_dir, source)
        try:
            df = read_csv_to_df(file_path)
            data[metric_name] = prepare_time_series(df)
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            continue  # Skip to the next file

    # Calculate derived metrics
    data["marcap"] = data.get("stock-price", pd.DataFrame()) * data.get(
        "shares-outstanding", pd.DataFrame()
    )
    data["ev"] = (
        data.get("marcap", pd.DataFrame())
        + data.get("total-debt", pd.DataFrame())
        - data.get("total-cash", pd.DataFrame())
    )
    data["debt_to_ebitda"] = data.get("total-debt", pd.DataFrame()) / data.get(
        "ebitda", pd.DataFrame()
    )
    data["debt_to_cfo"] = data.get("total-debt", pd.DataFrame()) / data.get(
        "cfo", pd.DataFrame()
    )
    data["marcap_to_sales"] = data.get("marcap", pd.DataFrame()) / data.get(
        "revenue", pd.DataFrame()
    )
    data["marcap_to_cfo"] = data.get("marcap", pd.DataFrame()) / data.get(
        "cfo", pd.DataFrame()
    )
    data["ev_to_ebitda"] = data.get("ev", pd.DataFrame()) / data.get(
        "ebitda", pd.DataFrame()
    )

    return data


def calculate_mean_minus_std(
    series: pd.Series,
    is_percentages: bool = False,
    last_quarters: int = 12,
    qoq: bool = False,
) -> float:
    """
    Calculate mean growth rate subtracted by std.
    This metric tells how high is the growth rate and how regular it is.

    Args:
        series (pd.Series): The time series data.
        is_percentages (bool): Whether the series represents percentages.
        last_quarters (int): The number of last quarters to consider.
        qoq (bool): Whether to calculate quarter-over-quarter changes.

    Returns:
        float: The mean growth rate subtracted by the standard deviation.
    """
    # Filter out consecutive duplicate values
    unique_vals = series[series.diff() != 0]

    # Calculate percentage change
    if is_percentages:
        perc_chg = (
            unique_vals.diff(periods=4)[-last_quarters:]
            if qoq
            else unique_vals.diff()[-last_quarters:]
        )
    else:
        perc_chg = (
            unique_vals.pct_change(periods=4)[-last_quarters:]
            if qoq
            else unique_vals.pct_change()[-last_quarters:]
        )

    # Calculate mean change
    mean_chg = perc_chg.mean()

    # Return mean change subtracted by half the standard deviation
    return mean_chg - perc_chg.std() / 2


def normalize_ewm_growth(
    series: pd.Series,
    timerange: int = DEFAULT_TIMERANGE,
    low: float = EWM_LOW,
    high: float = EWM_HIGH,
    alpha: float = DEFAULT_ALPHA,
) -> pd.Series:
    """
    Calculate exponential moving average and normalize it by a given value range.

    Args:
        series (pd.Series): The time series data.
        timerange (int): The time range to consider.
        low (float): The lower bound for normalization.
        high (float): The upper bound for normalization.
        alpha (float): Smoothing factor for EWM.

    Returns:
        pd.Series: The normalized exponential moving average.
    """
    # Filter out consecutive duplicate values
    unique_vals = series[-timerange:][series[-timerange:].diff() != 0]

    # Calculate percentage change
    perc_chg = unique_vals.pct_change()

    # Calculate exponential moving average
    ewm = perc_chg.ewm(alpha=alpha).mean()

    # Normalize the exponential moving average
    norm_ewm = (ewm - low) / (high - low)

    return norm_ewm


def normalize_growth_rate(
    series: pd.Series, timerange: int = 365, low: float = 0.0, high: float = 0.05
) -> pd.Series:
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
    timerange: int = DEFAULT_TIMERANGE,
    q_low: float = Q_LOW,
    q_high: float = Q_HIGH,
) -> float:
    """
    Normalize last datapoint to quantile band along the given timerange.

    Args:
        series (pd.Series): The time series data.
        timerange (int): The time range to consider.
        q_low (float): The lower quantile.
        q_high (float): The upper quantile.

    Returns:
        float: The normalized value.
    """
    # Calculate quantiles
    qh = series[-timerange:].quantile(q_high)
    ql = series[-timerange:].quantile(q_low)

    # Normalize the last data point
    return (series.iloc[-1] - ql) / (qh - ql)


def calculate_debt_dynamic_score(
    series: pd.Series,
    threshold: float = DEBT_THRESHOLD,
    smoothing: bool = True,
    n_quarters: int = N_QUARTERS,
) -> float:
    """
    Measure debt dynamic. 0 - good, 1 - bad.

    Args:
        series (pd.Series): The time series data.
        threshold (float): The threshold for high debt.
        smoothing (bool): Whether to use smoothing.
        n_quarters (int): The number of quarters to consider.

    Returns:
        float: The debt dynamic score.
    """
    # Check if debt is above the threshold
    above_thresh = series.iloc[-1] > threshold

    # Normalize the series
    if smoothing:
        series_norm = normalize_ewm_growth(
            series, 3 * DEFAULT_TIMERANGE, low=EWM_LOW, high=EWM_HIGH
        )
    else:
        series_norm = normalize_growth_rate(
            series, 3 * DEFAULT_TIMERANGE, low=EWM_LOW, high=EWM_HIGH
        )

    # Calculate the average normalized growth over the last n quarters
    qs = np.array([series_norm[-q:].mean() for q in range(n_quarters, 0, -1)])

    # Calculate the proportion of quarters where debt is under control
    duc_quarters = (qs <= 1).mean()

    # Calculate the debt down ratio
    epsilon = 0.1
    debt_down_points = sum(
        (qs[x] + epsilon >= qs[x + 1:]).sum() for x in range(n_quarters - 1)
    )
    debt_down_ratio = debt_down_points / 10

    # Combine the components to calculate the debt dynamic score
    return float(above_thresh) * (1 - (duc_quarters + debt_down_ratio) / 2)


def calculate_scores(ticker: str, data: dict) -> dict:
    """
    Calculate scores for a given ticker based on the provided data.

    Args:
        ticker (str): The ticker symbol.
        data (dict): A dictionary containing the data.

    Returns:
        dict: A dictionary containing the calculated scores.
    """
    out = {}

    # Growth metrics
    ebitda_mean_minus_std = calculate_mean_minus_std(
        data["ebitda"][ticker], last_quarters=8
    )
    ebitda_qoq_mean_minus_std = calculate_mean_minus_std(
        data["ebitda"][ticker], last_quarters=8, qoq=True
    )

    revenue_mean_minus_std = calculate_mean_minus_std(
        data["revenue"][ticker], last_quarters=8
    )
    revenue_qoq_mean_minus_std = calculate_mean_minus_std(
        data["revenue"][ticker], last_quarters=8, qoq=True
    )

    cfo_mean_minus_std = calculate_mean_minus_std(data["cfo"][ticker], last_quarters=8)
    cfo_qoq_mean_minus_std = calculate_mean_minus_std(
        data["cfo"][ticker], last_quarters=8, qoq=True
    )

    op_margin_mean_minus_std = calculate_mean_minus_std(
        data["operating-margin"][ticker], is_percentages=True, last_quarters=8
    )
    op_margin_qoq_mean_minus_std = calculate_mean_minus_std(
        data["operating-margin"][ticker], is_percentages=True, last_quarters=8, qoq=True
    )

    # Debt metrics
    debt_to_cfo_score = calculate_debt_dynamic_score(data["debt_to_cfo"][ticker])
    debt_to_ebitda_score = calculate_debt_dynamic_score(data["debt_to_ebitda"][ticker])

    # Valuation metrics
    ps_score = normalize_by_quantile_band(
        data["marcap_to_sales"][ticker], 3 * DEFAULT_TIMERANGE
    )
    p_cfo_score = normalize_by_quantile_band(
        data["marcap_to_cfo"][ticker], 3 * DEFAULT_TIMERANGE
    )
    ev_ebitda_score = normalize_by_quantile_band(
        data["ev_to_ebitda"][ticker], 3 * DEFAULT_TIMERANGE
    )

    # Growth metrics
    out["ebitda_growth_score"] = np.clip(ebitda_mean_minus_std, 0, EBITDA_CLIP)
    out["ebitda_qoq_growth_score"] = np.clip(ebitda_qoq_mean_minus_std, 0, EBITDA_CLIP)

    out["revenue_growth_score"] = np.clip(revenue_mean_minus_std, 0, EBITDA_CLIP)
    out["revenue_qoq_growth_score"] = np.clip(
        revenue_qoq_mean_minus_std, 0, EBITDA_CLIP
    )

    out["cfo_growth_score"] = np.clip(cfo_mean_minus_std, 0, EBITDA_CLIP)
    out["cfo_qoq_growth_score"] = np.clip(cfo_qoq_mean_minus_std, 0, EBITDA_CLIP)

    out["op_margin_growth_score"] = np.clip(
        op_margin_mean_minus_std, OP_MARGIN_CLIP_LOW, OP_MARGIN_CLIP_HIGH
    )
    out["op_margin_qoq_growth_score"] = np.clip(
        op_margin_qoq_mean_minus_std, OP_MARGIN_CLIP_LOW, OP_MARGIN_CLIP_HIGH
    )

    # Debt metrics
    out["debt_to_cfo_score"] = 1 - debt_to_cfo_score
    out["debt_to_ebitda_score"] = 1 - debt_to_ebitda_score

    # Valuation metrics
    out["ps_score"] = np.clip(1 - ps_score, 0, 1)
    out["p_cfo_score"] = np.clip(1 - p_cfo_score, 0, 1)
    out["ev_ebitda_score"] = np.clip(1 - ev_ebitda_score, 0, 1)

    return out


if __name__ == "__main__":
    # Load data
    data = load_and_prepare_data(DYNAMIC_METRICS_DIR, DATA_SOURCES)

    # Calculate scores for each ticker
    tickers_scores = {
        ticker: calculate_scores(ticker, data)
        for ticker in data.get("revenue", pd.DataFrame()).columns
    }

    # Create a DataFrame from the scores
    df_scores = pd.DataFrame.from_dict(tickers_scores)
    print(df_scores)

    # Plots
    # import matplotlib.pyplot as plt
    # data["marcap_to_sales"].plot.line()
    # plt.savefig("marcap_to_sales.png")
