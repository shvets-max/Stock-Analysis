import os
import logging
import pandas as pd

from constants import METRICS_DIR
from metrics import Forecasts, Growth, Valuation, Performance, Fundamentals

EXP = 2

logging.basicConfig(level=logging.INFO)

def calculate_scores():
    # df_general = pd.read_csv(os.path.join(METRICS_DIR, "general.csv"), index_col=0)

    # Forecasts
    forecasts = Forecasts()
    forecasts.load_csv(os.path.join(METRICS_DIR, "forecasts.csv"))
    forecasts.normalize_metrics()
    forecasts.initialize_weights()
    forecasts.calculate_weights()
    forecasts.calculate_scores()
    # forecasts.plot_sector_stats()

    # Valuation
    valuation = Valuation()
    valuation.load_csv(os.path.join(METRICS_DIR, "value.csv"))
    valuation.normalize_metrics(by="Sector")
    valuation.initialize_weights()
    valuation.calculate_scores()
    # valuation.plot_sector_stats()

    # Historical Growth
    growth = Growth()
    growth.load_csv(os.path.join(METRICS_DIR, "growth.csv"))
    growth.normalize_metrics()
    growth.initialize_weights()
    # growth.calculate_weights()
    growth.calculate_scores()
    # growth.plot_sector_stats()

    # Performance
    performance = Performance()
    performance.load_csv(os.path.join(METRICS_DIR, "performance.csv"))
    performance.normalize_metrics()
    performance.initialize_weights()
    # performance.calculate_weights()
    performance.calculate_scores()
    # performance.plot_sector_stats()

    # Fundamentals
    fundamentals = Fundamentals()
    fundamentals.load_csv(os.path.join(METRICS_DIR, "fundamentals.csv"))
    fundamentals.normalize_metrics(by="Sector")
    fundamentals.initialize_weights()
    # fundamentals.calculate_weights()
    fundamentals.calculate_scores()
    # fundamentals.plot_sector_stats()

    growth_score = (growth.normalized_data * growth.weights).sum(axis=1) / growth.weights.sum(axis=1)
    valuation_score = (valuation.normalized_data * valuation.weights).sum(axis=1) / valuation.weights.sum(axis=1)
    forecasts_score = (forecasts.normalized_data * forecasts.weights).sum(axis=1) / forecasts.weights.sum(axis=1)
    performance_score = (performance.normalized_data * performance.weights).sum(axis=1) / performance.weights.sum(axis=1)
    fundamentals_score = (fundamentals.normalized_data * fundamentals.weights).sum(axis=1) / fundamentals.weights.sum(
        axis=1)

    return growth_score, valuation_score, forecasts_score, performance_score, fundamentals_score

if __name__ == "__main__":
    grwth, value, forcst, perf, fund = calculate_scores()

    # Combine scores from less crucial (1/EXP power) to most crucial (EXP power).
    final_score = fund ** (1/EXP) * perf ** (1/EXP) * grwth * value**EXP * forcst**EXP
    final_score_sorted = final_score.sort_values(ascending=False)
    top_tickers = final_score_sorted[final_score_sorted > 1e-3]
    if top_tickers.size > 50:
        top_tickers = top_tickers[:50]

    with open("top50-static-scoring.txt", "w") as f:
        f.write("\n".join(top_tickers.index.tolist()))

