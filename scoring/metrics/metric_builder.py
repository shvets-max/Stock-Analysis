from abc import ABC, abstractmethod
import pandas as pd


class MetricBuilder(ABC):
    """
    Abstract class for metric groups.
    """
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



