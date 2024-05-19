"""Descriptive statistics functions"""

from typing import Callable
import numpy as np
from numpy.typing import NDArray
from scipy.stats import t, skew, kurtosis

from scigraph._options import SummaryStatistic


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
    def range(arr: NDArray) -> float:
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
        return skew(arr, nan_policy="omit")  # type: ignore

    @staticmethod
    def kurtosis(arr: NDArray) -> float:
        return kurtosis(arr, nan_policy="omit")  # type: ignore

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
        return critical_val * np.nanstd(arr) / n**0.5


type SummaryStatFn = Callable[[NDArray], float]


def get_summary_statistic_fn(stat: SummaryStatistic | str) -> SummaryStatFn:
    if isinstance(stat, str):
        stat = SummaryStatistic.from_str(stat)
    if stat in FN_MAP:
        return FN_MAP[stat]
    raise NotImplementedError


FN_MAP: dict[SummaryStatistic, SummaryStatFn] = {
    SummaryStatistic.MEAN: Basic.mean,
    SummaryStatistic.SD: Basic.sd,
    SummaryStatistic.SEM: Basic.sem,
    SummaryStatistic.SUM: Basic.sum,
    SummaryStatistic.MIN: Basic.min,
    SummaryStatistic.MAX: Basic.max,
    SummaryStatistic.RANGE: Basic.range,
    SummaryStatistic.N: Basic.n,
    SummaryStatistic.LOWER_QUARTILE: Basic.lower_quartile,
    SummaryStatistic.UPPER_QUARTILE: Basic.upper_quartile,
    SummaryStatistic.MEDIAN: Basic.median,
    SummaryStatistic.CV: Advanced.coefficient_of_variation,
    SummaryStatistic.SKEWNESS: Advanced.skewness,
    SummaryStatistic.KURTOSIS: Advanced.kurtosis,
    SummaryStatistic.GEOMETRIC_MEAN: Advanced.geometric_mean,
    SummaryStatistic.GEOMETRIC_SD: Advanced.geometric_sd,
    SummaryStatistic.MEAN_CI: ConfidenceInterval.mean,
}
