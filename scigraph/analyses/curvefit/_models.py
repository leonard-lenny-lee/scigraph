from __future__ import annotations

__all__ = [
    "Constant",
    "Linear",
    "Polynomial",
    "ExponentialDecay",
    "PiecewiseConstantExponentialDecay",
    "ExponentialGrowth",
    "Gaussian",
    "LogNormal",
    "Lorentzian",
    "Sinusoid",
    "Logistic3Parameter",
    "Logistic4Parameter",
    "Logistic5Parameter",
]

from functools import cached_property
from typing import Literal, override, TYPE_CHECKING

import numpy as np

from scigraph.analyses.curvefit._curvefit import CurveFit

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from scigraph.datatables import XYTable


class Constant(CurveFit):
    """Fit a constant response, ``c``."""

    @override
    @staticmethod
    def _f(x: NDArray, c: float) -> NDArray:  # type: ignore
        return c + x * 0


class Linear(CurveFit):
    """Fit a straight line, ``m * x + c``."""

    @override
    @staticmethod
    def _f(x: NDArray, m: float, c: float) -> NDArray:  # type: ignore
        return m * x + c


class Polynomial(CurveFit):
    """Fit a polynomial with coefficients ordered from constant to highest power."""

    def __init__(
        self,
        table: XYTable,
        order: int,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        confidence_level: float = 0.95,
        confidence_interval_method: Literal[
            "profile likelihood", "approximate", "bootstrap", "none"
        ] = "profile likelihood",
        bootstrap_samples: int = 1000,
    ) -> None:
        """Create a polynomial fit of the requested non-negative ``order``."""
        self._order = order
        super().__init__(
            table,
            include,
            exclude,
            confidence_level=confidence_level,
            confidence_interval_method=confidence_interval_method,
            bootstrap_samples=bootstrap_samples,
        )
        self._n_params = order + 1

    @cached_property
    @override
    def _params(self) -> tuple[str, ...]:
        return tuple([f"a{n}" for n in range(self._order + 1)])

    @override
    @staticmethod
    def _f(x: NDArray, *args: float) -> NDArray:
        out = np.zeros(len(x))
        for degree, coef in enumerate(args):
            out += coef * x**degree
        return out


class ExponentialDecay(CurveFit):
    """Fit exponential decay, ``c + (y0 - c) * exp(-k * x)``."""

    @override
    @staticmethod
    def _f(x: NDArray, y0: float, c: float, k: float) -> NDArray:  # type: ignore
        return c + (y0 - c) * np.exp(-k * x)


class PiecewiseConstantExponentialDecay(CurveFit):
    """Fit a constant response before ``x0`` and exponential decay afterwards."""

    @override
    @staticmethod
    def _f(x: NDArray, x0: float, y0: float, c: float, k: float) -> NDArray:  # type: ignore
        return np.piecewise(
            x, [x < x0, x >= x0], [y0, lambda x: c + (y0 - c) * np.exp(-k * (x - x0))]
        )


class ExponentialGrowth(CurveFit):
    """Fit exponential growth, ``y0 * exp(k * x)``."""

    @override
    @staticmethod
    def _f(x: NDArray, y0: float, k: float) -> NDArray:  # type: ignore
        return y0 * np.exp(k * x)


class Gaussian(CurveFit):
    """Fit a Gaussian peak with amplitude, centre, and standard deviation."""

    @override
    @staticmethod
    def _f(x: NDArray, a: float, m: float, s: float) -> NDArray:  # type: ignore
        return a * np.exp(-0.5 * ((x - m) / s) ** 2)


class LogNormal(CurveFit):
    """Fit a log-normal peak with amplitude, geometric mean, and geometric SD."""

    @override
    @staticmethod
    def _f(x: NDArray, a: float, gm: float, gsd: float) -> NDArray:  # type: ignore
        return (a / x) * np.exp(-0.5 * (np.log(x / gm) / np.log(gsd)) ** 2)


class Lorentzian(CurveFit):
    """Fit a Lorentzian peak with amplitude, centre, and half-width."""

    @override
    @staticmethod
    def _f(x: NDArray, a: float, x0: float, g: float) -> NDArray:  # type: ignore
        return a / (1 + ((x - x0) / g) ** 2)


class Sinusoid(CurveFit):
    """Fit a sinusoid with amplitude, frequency, phase, and offset."""

    @override
    @staticmethod
    def _f(x: NDArray, a: float, f: float, phase: float, c: float) -> NDArray:  # type: ignore
        return a * np.sin(2 * np.pi * f * x + phase) + c


class Logistic3Parameter(CurveFit):
    """Fit a three-parameter hyperbolic dose-response curve."""

    @override
    @staticmethod
    def _f(x: NDArray, top: float, bottom: float, ec50: float) -> NDArray:  # type: ignore
        return bottom + x * (top - bottom) / (ec50 + x)


class Logistic4Parameter(CurveFit):
    """Fit a four-parameter logistic dose-response curve."""

    @override
    @staticmethod
    def _f(
        x: NDArray, slope: float, top: float, bottom: float, ec50: float  # type: ignore
    ) -> NDArray:
        return bottom + (x**slope) * (top - bottom) / (x**slope + ec50**slope)


class Logistic5Parameter(CurveFit):
    """Fit an asymmetric five-parameter logistic dose-response curve."""

    @override
    @staticmethod
    def _f(
        x: NDArray,
        slope: float,
        top: float,
        bottom: float,  # type: ignore
        ec50: float,
        s: float,
    ) -> NDArray:
        return bottom + (
            (top - bottom) / ((1 + (2 ** (1 / s) - 1) * ((ec50 / x) ** slope)) ** s)
        )
