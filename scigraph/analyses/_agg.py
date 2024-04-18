import numpy as np
from numpy.typing import NDArray
from scipy.stats import t


class Basic:

    @staticmethod
    def mean(arr: NDArray) -> float:
        return np.nanmean(arr)  # type: ignore
    
    @staticmethod
    def sd(arr: NDArray) -> float:
        return np.nanstd(arr)  # type: ignore
    
    @staticmethod
    def sem(arr: NDArray) -> float:
        return np.nanstd(arr) / np.count_nonzero(arr) ** 0.5
    
    @staticmethod
    def sum(arr: NDArray) -> float:
        return np.nansum(arr)
    
    @staticmethod
    def min(arr: NDArray) -> float:
        return np.nanmin(arr)
    
    @staticmethod
    def max(arr: NDArray) -> float:
        return np.nanmax(arr)
    
    @staticmethod
    def range_(arr: NDArray) -> float:
        return np.nanmax(arr) - np.nanmin(arr)
    
    @staticmethod
    def n(arr: NDArray) -> int:
        return np.count_nonzero(arr)
    
    @staticmethod
    def lower_quartile(arr: NDArray) -> float:
        return np.nanquantile(arr, 0.25)  # type: ignore
    
    @staticmethod
    def median(arr: NDArray) -> float:
        return np.nanquantile(arr, 0.5)  # type: ignore
    
    @staticmethod
    def upper_quartile(arr: NDArray) -> float:
        return np.nanquantile(arr, 0.75)  # type: ignore
    
    @staticmethod
    def percentile(arr: NDArray, q: float) -> float:
        return np.nanquantile(arr, q)  # type: ignore
    

class Advanced:

    @staticmethod
    def coefficient_of_variation(arr: NDArray) -> float:
        return np.nanstd(arr) / np.nanmean(arr)  # type: ignore

    @staticmethod
    def skewness(arr: NDArray) -> float:
        """Calculate according to adjusted Fischer-Pearson coefficient"""
        n = np.count_nonzero(arr)
        m_3 = np.nansum(np.power((arr - np.nanmean(arr)), 3)) / n
        b_1 = m_3 / np.nanstd(arr) ** 3
        return (((n * (n - 1)) ** 0.5) / (n - 2)) * b_1

    @staticmethod
    def geometric_mean(arr: NDArray) -> float:
        return 10 ** np.nanmean(np.log10(arr))  # type: ignore

    @staticmethod
    def geometric_sd(arr: NDArray) -> float:
        return 10 ** np.nanstd(np.log10(arr))  # type: ignore


class ConfidenceInterval:

    @staticmethod
    def mean(arr: NDArray, level: float = 0.95) -> float:
        n = np.count_nonzero(arr)
        critical_val = t.ppf((1 + level) / 2, n - 1)
        return critical_val * np.nanstd(arr) / n ** 0.5
