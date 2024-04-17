from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t


class Basic:

    @staticmethod
    def mean(arr: NDArray) -> float:
        return arr.mean()
    
    @staticmethod
    def sd(arr: NDArray) -> float:
        return arr.std()
    
    @staticmethod
    def sem(arr: NDArray) -> float:
        return arr.std() / np.count_nonzero(arr) ** 0.5
    
    @staticmethod
    def sum(arr: NDArray) -> float:
        return arr.sum()
    
    @staticmethod
    def min(arr: NDArray) -> float:
        return arr.min()
    
    @staticmethod
    def max(arr: NDArray) -> float:
        return arr.max()
    
    @staticmethod
    def range_(arr: NDArray) -> float:
        return arr.max() - arr.min()
    
    @staticmethod
    def n(arr: NDArray) -> int:
        return np.count_nonzero(arr)
    
    @staticmethod
    def lower_quartile(arr: NDArray) -> float:
        return np.quantile(arr, 0.25)  # type: ignore
    
    @staticmethod
    def median(arr: NDArray) -> float:
        return np.quantile(arr, 0.5)  # type: ignore
    
    @staticmethod
    def upper_quartile(arr: NDArray) -> float:
        return np.quantile(arr, 0.75)  # type: ignore
    
    @staticmethod
    def percentile(arr: NDArray, q: float) -> float:
        return np.quantile(arr, q)  # type: ignore
    

class Advanced:

    @staticmethod
    def coefficient_of_variation(arr: NDArray) -> float:
        return arr.std() / arr.mean()

    @staticmethod
    def skewness(arr: NDArray) -> float:
        """Calculate according to adjusted Fischer-Pearson coefficient"""
        n = np.count_nonzero(arr)
        m_3 = np.sum(np.power((arr - arr.mean()), 3)) / n
        b_1 = m_3 / arr.std() ** 3
        return (((n * (n - 1)) ** 0.5) / (n - 2)) * b_1

    @staticmethod
    def geometric_mean(arr: NDArray) -> float:
        return 10 ** np.log10(arr).mean()

    @staticmethod
    def geometric_sd(arr: NDArray) -> float:
        return 10 ** np.log10(arr).std()


class ConfidenceInterval:

    @staticmethod
    def mean(arr: NDArray, level: float = 0.95) -> float:
        n = np.count_nonzero(arr)
        critical_val = t.ppf((1 + level) / 2, n - 1)
        return critical_val * arr.std() / n ** 0.5


_MAP = {
    "mean": Basic.mean,
    "median": Basic.median,
}


def agg(s: str) -> Callable[..., float]:
    return _MAP[s]
