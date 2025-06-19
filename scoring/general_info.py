import os

import pandas as pd

from scoring.constants import METRICS_DIR


class GeneralInfo:
    def __init__(self):
        self.__data = None

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, value: pd.DataFrame):
        self.__data = value

    def load_csv(self):
        self.data = pd.read_csv(os.path.join(METRICS_DIR, "general.csv"), index_col=0)
