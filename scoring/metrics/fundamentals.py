import logging
from typing import List

import numpy as np
import pandas as pd

from scoring.constants import ALLOWED_GROUPS, Q_LOW, Q_HIGH, MIN_GROUP
from scoring.metrics import MetricBuilder, GeneralInfo
from utils.static import load_and_normalize_percentages

FUNDAMENTALS_WEIGHTS = {
    "Quick Ratio": 2,
    "Debt / EBITDA": 3,
    "Debt / FCF": 1,
    "Debt / Equity": 1,
    "Current Ratio": 2
}
LOWER_PREFERRED = ["Debt / EBITDA", "Debt / FCF", "Debt / Equity"]

geninfo = GeneralInfo()
geninfo.load_csv()


class Fundamentals(MetricBuilder):
    def __init__(self):
        self.__data = None
        self.__normalized_data = None
        self.__weights = None
        self.__scores = None

    @property
    def metric_columns(self) -> List:
        return list(FUNDAMENTALS_WEIGHTS.keys())

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, value: pd.DataFrame):
        self.__data = value

    @property
    def normalized_data(self):
        return self.__normalized_data

    @normalized_data.setter
    def normalized_data(self, value: pd.DataFrame):
        self.__normalized_data = value

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, value: pd.DataFrame):
        self.__weights = value

    @property
    def scores(self):
        return self.__scores

    @scores.setter
    def scores(self, value: pd.DataFrame):
        self.__scores = value

    def load_csv(self, csv_path: str):
        _data = load_and_normalize_percentages(csv_path, norm_columns=self.metric_columns)
        self.data = _data.merge(geninfo.data, on="Symbol")

    @property
    def quantiles(self):
        stats = {"global": {
            "q_low": self.data[self.metric_columns].quantile(Q_LOW),
            "q_high": self.data[self.metric_columns].quantile(Q_HIGH),
            "counts": self.data.shape[0]
        }}
        for key in ALLOWED_GROUPS:
            g = self.data[[key] + self.metric_columns].groupby(key)
            q_low = g.quantile(Q_LOW)
            q_high = g.quantile(Q_HIGH)
            counts = g.count()
            min_bools = (counts < MIN_GROUP).mean(axis=1) > 0.5

            if np.any(min_bools):
                minorities = counts.index[min_bools].tolist()
                logging.warning(
                    f"{self.__class__.__name__}: Not enough peers to calculate quantiles in {key} = {minorities}. "
                    f"Global quantiles will be assigned for these groups."
                )

                # Assign global quantiles for minority groups
                q_low.loc[minorities, self.metric_columns] = pd.DataFrame({col: stats["global"]["q_low"] for col in minorities}).T
                q_high.loc[minorities, self.metric_columns] = pd.DataFrame({col: stats["global"]["q_high"] for col in minorities}).T

            stats[key] = {
                "q_low": q_low,
                "q_high": q_high,
                "counts": counts
            }
        return stats

    def normalize_metrics(
            self,
            by: str = None,
    ):
        _data = self.data.copy()

        if by is not None and by in self.quantiles:
            qs_low = self.quantiles[by]["q_low"]
            qs_high = self.quantiles[by]["q_high"]
            non_na = ~_data[by].isna()

            for col in self.metric_columns:
                ql = qs_low.loc[_data.loc[non_na, by], col].values
                qh = qs_high.loc[_data.loc[non_na, by], col].values
                _data.loc[non_na, col] = np.clip((_data.loc[non_na, col] - ql) / (qh - ql), 0, 1)

                _ql = self.quantiles["global"]["q_low"]
                _qh = self.quantiles["global"]["q_high"]
                _data.loc[~non_na, col] = np.clip((_data.loc[~non_na, col] - _ql) / (_qh - _ql), 0, 1)

        else:
            _data[self.metric_columns] = _data[self.metric_columns].apply(
                lambda x: np.clip(
                    (x - x.quantile(Q_LOW)) / (
                            x.quantile(Q_HIGH) - x.quantile(Q_LOW)
                    ),
                    0, 1
                )
            )

        # Reverse values if low preferred.
        _data[LOWER_PREFERRED] = 1 - _data[LOWER_PREFERRED]

        _data.fillna(0, inplace=True)
        self.normalized_data = _data

    def initialize_weights(self):
        self.weights = pd.DataFrame(
            data={k: [v] * self.data.shape[0] for k, v in FUNDAMENTALS_WEIGHTS.items()},
            index=self.data.index
        )

    def calculate_weights(self):
        pass

    def calculate_scores(self):
        metric_names = self.weights.columns
        total_weights = self.weights.sum(axis=1)
        _weighted = self.normalized_data[metric_names] * self.weights
        _scores = _weighted.sum(axis=1)
        _norm_scores = _scores / total_weights

        self.scores = pd.DataFrame(
            {
                "Scores": _scores.tolist(),
                "Normalized": _norm_scores.tolist()
            },
            index=self.normalized_data.index
        )
