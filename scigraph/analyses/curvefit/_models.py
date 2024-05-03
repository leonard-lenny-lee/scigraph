from __future__ import annotations

__all__ = ["Constant", "Linear", "Polynomial", "ExponentialDecay",
           "PiecewiseConstantExponentialDecay", "ExponentialGrowth",
           "Gaussian", "Lorentzian", "Sinusoid", "Logistic3Parameter",
           "Logistic4Parameter", "Logistic5Parameter"]

from typing import override, TYPE_CHECKING

import numpy as np

from scigraph.analyses.curvefit._curvefit import CurveFit

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from scigraph.datatables import XYTable


class Constant(CurveFit):

    @override
    @staticmethod
    def _f(x: NDArray, c: float) -> NDArray:  # type: ignore
        return c + x * 0


class Linear(CurveFit):

    @override
    @staticmethod
    def _f(x: NDArray, m: float, c: float) -> NDArray:  # type: ignore
        return m * x + c


class Polynomial(CurveFit):

    def __init__(
        self,
        table: XYTable,
        order: int,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        confidence_level: float = 0.95,
    ) -> None:
        self._order = order
        super().__init__(table, include, exclude, confidence_level)

    @override
    def _params(self) -> tuple[str, ...]:
        return tuple([f"a{n}" for n in range(self._order + 1)])

    @property
    @override
    def _n_params(self) -> int:
        return self._order + 1

    @override
    @staticmethod
    def _f(x: NDArray, *args: float) -> NDArray:
        out = np.zeros(len(x))
        for degree, coef in enumerate(args):
            out += coef * x ** degree
        return out


class ExponentialDecay(CurveFit):

    @override
    @staticmethod
    def _f(x: NDArray, y0: float, c: float, k: float) -> NDArray:  # type: ignore
        return c + (y0 - c) * np.exp(-k * x)


class PiecewiseConstantExponentialDecay(CurveFit):

    @override
    @staticmethod
    def _f(x: NDArray, x0: float, y0: float, c: float, k: float) -> NDArray:  # type: ignore
        return np.piecewise(x, [x < x0, x >= x0],
                            [y0, lambda x: c + (y0 - c) * np.exp(-k * (x - x0))])


class ExponentialGrowth(CurveFit):

    @override
    @staticmethod
    def _f(x: NDArray, y0: float, k: float) -> NDArray:  # type: ignore
        return y0 * np.exp(k * x)


class Gaussian(CurveFit):

    @override
    @staticmethod
    def _f(x: NDArray, a: float, m: float, s: float) -> NDArray:  # type: ignore
        return a * np.exp(-0.5 * ((x - m) / s) ** 2)


class LogNormal(CurveFit):

    @override
    @staticmethod
    def _f(x: NDArray, a: float, gm: float, gsd: float) -> NDArray:  # type: ignore
        return (a / x) * np.exp(-0.5 * (np.log(x / gm) / np.log(gsd)) ** 2)


class Lorentzian(CurveFit):

    @override
    @staticmethod
    def _f(x: NDArray, a: float, x0: float, g: float) -> NDArray:  # type: ignore
        return a / (1 + ((x - x0) / g) ** 2)


class Sinusoid(CurveFit):

    @override
    @staticmethod
    def _f(x: NDArray, a: float, f: float, phase: float, c: float) -> NDArray:  # type: ignore
        return a * np.sin(2 * np.pi * f * x + phase) + c


class Logistic3Parameter(CurveFit):

    @override
    @staticmethod
    def _f(x: NDArray, top: float, bottom: float, ec50: float) -> NDArray:  # type: ignore
        return bottom + x * (top - bottom) / (ec50 + x)


class Logistic4Parameter(CurveFit):

    @override
    @staticmethod
    def _f(x: NDArray, slope: float, top: float, bottom: float,  # type: ignore
           ec50: float) -> NDArray:
        return bottom + (x ** slope) * (top - bottom) / (x ** slope + ec50 ** slope)


class Logistic5Parameter(CurveFit):

    @override
    @staticmethod
    def _f(x: NDArray, slope: float, top: float, bottom: float,  # type: ignore
           ec50: float, s: float) -> NDArray:
        return bottom + ((top - bottom) / ((1 + (2 ** (1 / s) - 1) * ((ec50 / x) ** slope)) ** s))
