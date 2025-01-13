import numpy as np


def sigma(series):
    mean = np.mean(series)
    diff = abs(series - mean)
    return np.mean(diff ** 2) ** (1/2)

def sigma_adj(series):
    slope = (series[-1] - series[0]) / len(series)
    slope_adj_line = [series[0] + i * slope for i in range(len(series))]
    diff = abs(series - slope_adj_line)
    return np.mean(diff ** 2) ** (1 / 2)


def VaR():
    """
    Value at risk.
    :return:
    """
    pass

def ES():
    """
    Expected shortfall
    :return:
    """
    pass