from abc import ABC, abstractmethod
from typing import override, Any

import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes
from scipy.stats import t, gstd

from ..abc import GraphTypeCheckComponent


class ErrorBars(GraphTypeCheckComponent, ABC):

    @classmethod
    @abstractmethod
    def plot_group(
        cls,
        x: NDArray,
        y: NDArray,
        ax: Axes,
        xerr: NDArray,
        yori: NDArray,
        *args,
        **kwargs,
    ) -> None:
        yerr = cls._prepare_yerr(y, yori)
        ax.errorbar(x, yori, xerr=xerr, yerr=yerr,
                    *args, **kwargs, marker="", ls="")

    @classmethod
    @abstractmethod
    def _prepare_yerr(cls, y: NDArray, ori: NDArray) -> NDArray: ...


class SDErrorBars(ErrorBars):

    TYPES = {"mean"}

    @override
    @classmethod
    def _prepare_yerr(cls, y: NDArray, _: Any) -> NDArray:
        return y.std(axis=1)


class SEMErrorBars(ErrorBars):

    TYPES = {"mean"}

    @override
    @classmethod
    def _prepare_yerr(cls, y: NDArray, _: Any) -> NDArray:
        count = np.count_nonzero(~np.isnan(y), axis=1, keepdims=True)
        return y.std(axis=1) / np.sqrt(count)


class CI95ErrorBars(ErrorBars):

    TYPES = {"mean", "geometric mean", "median"}

    @override
    @classmethod
    def _prepare_yerr(cls, y: NDArray, _: Any) -> NDArray:
        n = np.count_nonzero(~np.isnan(y), axis=1, keepdims=True)
        critical_val = t.ppf(0.975, n - 1)
        return critical_val * y.std(axis=1) / np.sqrt(n)


class RangeErrorBars(ErrorBars):

    TYPES = {"mean", "median"}

    @override
    @classmethod
    def _prepare_yerr(cls, y: NDArray, ori: NDArray) -> NDArray:
        lower = ori - y.min(axis=1)
        upper = y.max(axis=1) - ori
        return lower, upper


class GeometricSDErrorBars(ErrorBars):

    TYPES = {"geometric mean"}

    @override
    @classmethod
    def _prepare_yerr(cls, y: NDArray, _: Any) -> NDArray:
        return gstd(axis=1)


_FACTORY_MAP = {
    "sd": SDErrorBars,
    "sem": SEMErrorBars,
    "ci95": CI95ErrorBars,
    "range": RangeErrorBars,
    "geometric sd": GeometricSDErrorBars,
}


def errorbar_factory_fn(arg: str) -> type[ErrorBars] | None:
    if arg in _FACTORY_MAP:
        return _FACTORY_MAP[arg]
    return None
