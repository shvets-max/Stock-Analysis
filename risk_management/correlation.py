import os

from scoring.constants import METRICS_DIR
from scoring.utils import prepare_time_series, read_csv_to_df

DYNAMIC_METRICS_DIR = os.path.join(METRICS_DIR, "dynamic_metrics")


marcap = prepare_time_series(read_csv_to_df(os.path.join(DYNAMIC_METRICS_DIR, "marcap.csv")))
corr_2q = marcap[:-180].corr()
corr_1y = marcap[:-365].corr()
corr_3y = marcap[:-1100].corr()

std_corr = (4 * corr_2q + 2 * corr_1y + corr_3y) / 7
print(std_corr)