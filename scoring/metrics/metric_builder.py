from abc import ABC, abstractmethod
from typing import List

import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from string import punctuation, whitespace


class MetricBuilder(ABC):
    """
    Abstract class for metric groups.
    """
    @property
    @abstractmethod
    def metric_columns(self) -> List:
        ...

    @property
    @abstractmethod
    def data(self) -> pd.DataFrame:
        ...

    @data.setter
    @abstractmethod
    def data(self, value: pd.DataFrame):
        ...

    @property
    @abstractmethod
    def normalized_data(self) -> pd.DataFrame:
        return ...

    @normalized_data.setter
    @abstractmethod
    def normalized_data(self, value: pd.DataFrame):
        ...

    @property
    @abstractmethod
    def weights(self) -> pd.DataFrame:
        ...

    @weights.setter
    @abstractmethod
    def weights(self, value: pd.DataFrame):
        ...

    @property
    @abstractmethod
    def scores(self) -> pd.DataFrame:
        ...

    @scores.setter
    @abstractmethod
    def scores(self, value: pd.DataFrame):
        ...

    @abstractmethod
    def load_csv(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def normalize_metrics(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def calculate_weights(self):
        raise NotImplementedError

    @abstractmethod
    def calculate_scores(self):
        raise NotImplementedError

    def plot_sector_stats(self):
        plot_dir = os.path.join("plots", self.__class__.__name__)
        os.makedirs(plot_dir, exist_ok=True)
        plt.clf()

        for col in self.metric_columns:
            grouped_data = [self.data[self.data['Sector'] == sector][col] for sector in self.data['Sector'].unique()]
            grouped_data = [s[(-1.5 < s) & (s < 1.5)] for s in grouped_data]

            plt.boxplot(grouped_data, labels=self.data['Sector'].unique())
            plt.ylabel(col)
            plt.grid(axis='y')
            plt.xticks(rotation=90)
            plt.xlabel('Sector')
            plt.tick_params(axis='x', direction='in', pad=25)

            metric_name = re.sub(rf"[{punctuation + whitespace}]", "-", col)
            metric_name = re.sub(r"-+", "-", metric_name)
            plt.savefig(f"{plot_dir}/{metric_name}.png")
            plt.clf()
