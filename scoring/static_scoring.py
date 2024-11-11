import os
import logging
import pandas as pd

from constants import METRICS_DIR
from metrics import Forecasts, Growth, Valuation, Performance, Fundamentals

logging.basicConfig(level=logging.INFO)

df_general = pd.read_csv(os.path.join(METRICS_DIR, "general.csv"), index_col=0)

# Forecasts
forecasts = Forecasts()
forecasts.load_csv(os.path.join(METRICS_DIR, "forecasts.csv"))
forecasts.normalize_metrics()
forecasts.initialize_weights()
forecasts.calculate_weights()
forecasts.calculate_scores()

# Valuation
valuation = Valuation()
valuation.load_csv(os.path.join(METRICS_DIR, "value.csv"))
valuation.normalize_metrics(by="Sector")
valuation.initialize_weights()
valuation.calculate_scores()

# Historical Growth
growth = Growth()
growth.load_csv(os.path.join(METRICS_DIR, "growth.csv"))
growth.normalize_metrics()
growth.initialize_weights()
# growth.calculate_weights()
growth.calculate_scores()

# Performance
performance = Performance()
performance.load_csv(os.path.join(METRICS_DIR, "performance.csv"))
performance.normalize_metrics()
performance.initialize_weights()
# performance.calculate_weights()
performance.calculate_scores()

# Fundamentals
fundamentals = Fundamentals()
fundamentals.load_csv(os.path.join(METRICS_DIR, "fundamentals.csv"))
fundamentals.normalize_metrics()
fundamentals.initialize_weights()
# fundamentals.calculate_weights()
fundamentals.calculate_scores()


all_metrics_norm = pd.concat(
    [
        fundamentals.normalized_data,
        performance.normalized_data,
        growth.normalized_data,
        valuation.normalized_data,
        forecasts.normalized_data,
    ],
    axis=1
)
all_metrics_weights = pd.concat(
    [
        fundamentals.weights,
        performance.weights,
        growth.weights,
        valuation.weights,
        forecasts.weights,
    ],
    axis=1
)

final_score_norm = (all_metrics_norm * all_metrics_weights).sum(axis=1) / all_metrics_weights.sum(axis=1)


# top50_tickers = final_score_norm.sort_values(by="final_score", ascending=False).head(50).index.tolist()

# with open("top50-initial-scoring.txt", "w") as f:
#     f.write("\n".join(top50_tickers))
#
# Correlations
# corr = {"Market Cap": len(
#     set(df_general.sort_values(by="Market Cap", ascending=False).head(
#         75).index).intersection(set(top50_tickers))
# ) / 50}
#
# for metric in all_metrics_norm.columns:
#     top75by_metric = set(all_metrics_norm[metric].sort_values(ascending=False).head(75).index)
#     corr[metric] = len(top75by_metric.intersection(set(top50_tickers))) / 50


# ---------------
# Scoring correlations with different metrics_data
# import matplotlib.pyplot as plt
#
# sorted_weights = {k: v for k, v in sorted(corr.items(), key=lambda item: item[1], reverse=True)}
# labels, values = sorted_weights.keys(), sorted_weights.values()
# indexes = np.arange(len(labels))
#
# plt.bar(indexes, values)
# plt.xticks(indexes, labels, rotation=90)
# plt.title("Top stocks coverage by metric (50 in 75)")
# plt.show()

