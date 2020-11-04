
"""
Hurst exponent and RS-analysis
https://github.com/Mottl/hurst
"""

import math
import warnings
import numpy as np
import pandas as pd


def __to_inc(x):
    incs = x[1:] - x[:-1]
    return incs

def __to_pct(x):
    pcts = x[1:] / x[:-1] - 1.
    return pcts

def __get_simplified_RS(series, kind):
    """Simplified version of rescaled range

    Args:
        series (array-like): Time Series
        kind (str): The kind of series (refer to compute_Hc docstring)

    Returns:
        [float]: R/S ratio
    """

    if kind == 'random_walk':
        incs = __to_inc(series)
        R = max(series) - min(series)  # range in absolute values
        S = np.std(incs, ddof=1)
    elif kind == 'price':
        pcts = __to_pct(series)
        R = max(series) / min(series) - 1. # range in percent
        S = np.std(pcts, ddof=1)
    elif kind == 'change':
        incs = series
        _series = np.hstack([[0.],np.cumsum(incs)])
        R = max(_series) - min(_series)  # range in absolute values
        S = np.std(incs, ddof=1)

    if R == 0 or S == 0:
        return 0  # return 0 to skip this interval due the undefined R/S ratio

    return R / S

def __get_RS(series, kind):
    """
    Get rescaled range (using the range of cumulative sum
    of deviations instead of the range of a series as in the simplified version
    of R/S) from a time-series of values.

    Args:
        series (array-like): Time Series
        kind (str): The kind of series (refer to compute_Hc docstring)

    Returns:
        [float]: R/S ratio
    """

    if kind == 'random_walk':
        incs = __to_inc(series)
        mean_inc = (series[-1] - series[0]) / len(incs)
        deviations = incs - mean_inc
        Z = np.cumsum(deviations)
        R = max(Z) - min(Z)
        S = np.std(incs, ddof=1)
    elif kind == 'price':
        incs = __to_pct(series)
        mean_inc = np.sum(incs) / len(incs)
        deviations = incs - mean_inc
        Z = np.cumsum(deviations)
        R = max(Z) - min(Z)
        S = np.std(incs, ddof=1)
    elif kind == 'change':
        incs = series
        mean_inc = np.sum(incs) / len(incs)
        deviations = incs - mean_inc
        Z = np.cumsum(deviations)
        R = max(Z) - min(Z)
        S = np.std(incs, ddof=1)

    if R == 0 or S == 0:
        return 0  # return 0 to skip this interval due undefined R/S

    return R / S

def compute_Hc(series, kind="price", min_window=10, max_window=None, simplified=True):
    """
    Compute H (Hurst exponent) and C according to Hurst equation:
    E(R/S) = c * T^H

    Refer to:
        https://en.wikipedia.org/wiki/Hurst_exponent
        https://en.wikipedia.org/wiki/Rescaled_range
        https://en.wikipedia.org/wiki/Random_walk

    Args:
        series (array-like): Time Series
        kind (str, optional): Kind of series. Defaults to "random_walk".
            'random_walk' means that a series is a random walk with random increments
            'price' means that a series is a random walk with random multipliers
            'change' means that a series consists of random increments,
                thus produced random walk is a cumulative sum of increments
        min_window (int, optional): the minimal window size for R/S calculation. Defaults to 10.
        max_window (int, optional): the maximal window size for R/S calculation. Defaults to the length of series minus 1.
        simplified (bool, optional): whether to use the simplified or the original version of R/S calculation. Defaults to True.

    Raises:
        ValueError: Series length must be greater or equal to 100
        ValueError: Series contains NaNs

    Returns:
        H, c and data
        where H and c — parameters or Hurst equation
        and data is a list of 2 lists: time intervals and R/S-values for correspoding time interval
        for further plotting log(data[0]) on X and log(data[1]) on Y
    """
    series = np.array(series)

    if len(series) < 100:
        raise ValueError("Series length must be greater or equal to 100")
    if np.isnan(np.min(series)):
        raise ValueError("Series contains NaNs")

    RS_func = __get_simplified_RS if simplified else __get_RS

    # Set how floating-point errors are handled
    err = np.geterr()
    np.seterr(all='raise')

    max_window = max_window or (len(series) - 1)
    window_sizes = list(map(
        lambda x: int(10**x),
        np.arange(math.log10(min_window), math.log10(max_window), 0.25)))
    window_sizes.append(len(series))

    RS = []
    for w in window_sizes:
        rs = []
        for start in range(0, len(series), w):
            if (start+w) > len(series):
                break
            _ = RS_func(series[start:start+w], kind)
            if _ != 0:
                rs.append(_)
        RS.append(np.mean(rs))

    A = np.vstack([np.log10(window_sizes), np.ones(len(RS))]).T
    H, c = np.linalg.lstsq(A, np.log10(RS), rcond=-1)[0]
    np.seterr(**err)

    c = 10**c
    return H, c, [window_sizes, RS]

def random_walk(length, proba=0.5, min_lookback=1, max_lookback=100, cumprod=False):
    """Generates a random walk series

    Args:
        length (int): series length
        proba (float, optional): the probability that the next increment will follow the trend.
            Set proba > 0.5 for the persistent random walk,
            set proba < 0.5 for the antipersistent one
            Defaults to 0.5.

        min_lookback (int, optional): minimum window sizes to calculate trend direction.
            Defaults to 1.
        max_lookback (int, optional): maximum window sizes to calculate trend direction.
            Defaults to 100.
        cumprod (bool, optional): generate a random walk as a cumulative product instead of cumulative sum.
            Defaults to False.

    Returns:
        [array]: random walk series
    """

    assert (min_lookback >= 1)
    assert (max_lookback >= min_lookback)

    if max_lookback > length:
        max_lookback = length
        warnings.warn("max_lookback parameter has been set to the length of the random walk series.")

    if not cumprod:  # ordinary increments
        series = [0.] * length  # array of prices
        for i in range(1, length):
            if i < min_lookback + 1:
                direction = np.sign(np.random.randn())
            else:
                lookback = np.random.randint(min_lookback, min(i-1, max_lookback)+1)
                direction = np.sign(series[i-1] - series[i-1-lookback]) * np.sign(proba - np.random.uniform())
            series[i] = series[i-1] + np.fabs(np.random.randn()) * direction
    else:  # percent changes
        series = [1.] * length  # array of prices
        for i in range(1, length):
            if i < min_lookback + 1:
                direction = np.sign(np.random.randn())
            else:
                lookback = np.random.randint(min_lookback, min(i-1, max_lookback)+1)
                direction = np.sign(series[i-1] / series[i-1-lookback] - 1.) * np.sign(proba - np.random.uniform())
            series[i] = series[i-1] * np.fabs(1 + np.random.randn()/1000. * direction)

    return series
