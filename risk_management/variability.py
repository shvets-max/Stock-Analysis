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


def VaR(series, method="historical", conditional: bool = False):
    """

    :param series:
    :param method: 'historical' or 'std'
    :param conditional: True if CVaR (aka expected shortfall) should be returned
    :return:
    """
    if method not in ("historical", "std"):
        return None

    daily_returns = np.diff(series) / series

    if method == "std":
        exp_returns = np.mean(daily_returns)
        std_returns = np.std(daily_returns)
        var = exp_returns - 1.65 * std_returns
    else:
        var = np.quantile(daily_returns, q=.05)

    if conditional:
        return np.mean(daily_returns[daily_returns < var])

    return var
