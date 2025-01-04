import datetime
import os
import csv
import pandas as pd

from constants import METRICS_DIR

METRICS_WEIGHTS = {
    "Total Debt": 0,
    "Total Cash": 0,
    "CFO": 0,
    "Revenue": 0,
    "Marcap": 0,

    "EBITDA": 5,
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

# Debt metrics:
last_1Y_chg = data["debt_to_cfo"]["$MRK"][-400:][np.where(data["debt_to_cfo"]["$MRK"][-400:].diff() != 0)[0]]
q_gt05 = ((last_1Y_chg - last_1Y_chg.shift()) / last_1Y_chg > 0.05).mean()  # quarters with debt/cfo growth > 5%


# Valuation metrics:
q80 = data["marcap_to_sales"]["$MRK"].rolling(3*365).quantile(.8)[-1]
q20 = data["marcap_to_sales"]["$MRK"].rolling(3*365).quantile(.2)[-1]
last_val = data["marcap_to_sales"]["$MRK"][-1]
norm_val = (last_val - q20) / (q80 - q20) # normalized by 3Y trailing quantile band

# Plots
# import matplotlib.pyplot as plt
# data["marcap_to_sales"].plot.line()
# plt.savefig("marcap_to_sales.png")

