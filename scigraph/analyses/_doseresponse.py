from abc import ABC, abstractmethod
from typing import List

from numpy import log, exp, tile, linspace, min, max, sum
from pandas import DataFrame
import scipy.optimize as opt

from ..tables import XYTable
from ..graphing import LineGraph
from .._utils.args import Arg

__all__ = ["LL4", "DoseResponse"]


class DoseResponseEquation(ABC):

    @abstractmethod
    def f(self, x: float, *args, **kwargs) -> float: ...

    @property
    @abstractmethod
    def coef(self) -> List[str]: ...


class LL4(DoseResponseEquation):

    def f(self, x: float, b: float, c: float, d: float, e: float) -> float:
        """4 parameter logistic function with
        - x: dose
        - b: Hill slope
        - c: min response
        - d: max response
        - e: EC50
        """
        return c+((d-c)/(1+exp(b*(log(x)-log(e)))))

    @property
    def coef(self) -> List[str]:
        return ["b", "c", "d", "e"]


class HillEquation(DoseResponseEquation):

    def f(self, x: float, e_max: float, ec50: float, n: float) -> float:
        return e_max/(1+(ec50/exp(x))**n)

    @property
    def coef(self) -> List[str]:
        return ["e_max", "ec50", "n"]


class _Model(Arg):

    LL4 = LL4
    HILL = HillEquation


class DoseResponse:

    def __init__(self, dt: XYTable) -> None:
        self.dt = dt
        self._results = None

    def solve(self, model: str | DoseResponseEquation = "ll4"):
        self.model: DoseResponseEquation = model
        n = self.dt.n_y_replicates
        x = tile(self.dt.mean.values.T[0], n)
        fit_data = []
        for group in self.dt.groupnames:
            y = self.dt.data[group].values.T.ravel()
            fit_coef, _ = opt.curve_fit(self.model.f, x, y)
            residuals = sum((y - self.model.f(x, *fit_coef)) ** 2)
            cur_fit = dict(zip(self.model.coef, fit_coef))
            cur_fit["compound"] = group
            cur_fit["residuals"] = residuals
            fit_data.append(cur_fit)
        self._results = DataFrame(fit_data).set_index("compound")
        # TODO - add standard error on the regression analysis
        return self.results

    def graph(self, n_points: int = 1000) -> LineGraph:
        graph = LineGraph(self.dt, marker="o", linestyle="none")
        graph.plot()
        # Find ranges of x values to plot
        x_data = self.dt.data[self.dt.x_name].values
        x_min, x_max = min(x_data), max(x_data)
        x_buffer = (x_max - x_min) * 0.1
        x_min, x_max = x_min - x_buffer, x_max + x_buffer
        x = linspace(x_min, x_max, n_points)
        fit_coef = self.results[self.model.coef]
        for compound, *fit_coef in fit_coef.itertuples():
            y = self.model.f(x, *fit_coef)
            graph.axes.plot(x, y, label=compound)
            # TODO - implement label synchronization across elements
            # TODO - add max, min responses, EC50 with lines
        return graph

    @property
    def results(self) -> DataFrame:
        if self._results is None:
            raise NameError("call solve() before accessing results")
        return self._results

    @property
    def model(self) -> DoseResponseEquation:
        return self._model

    @model.setter
    def model(self, val: str | DoseResponseEquation) -> None:
        if isinstance(val, DoseResponseEquation):
            self._model = val
        else:
            self._model = _Model.from_str(val).value()
