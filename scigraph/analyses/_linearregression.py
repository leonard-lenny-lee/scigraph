"""Contains SimpleLinearRegression
"""
__all__ = ["SimpleLinearRegression"]

from numpy import tile, ndarray, array, linspace, min, max
from pandas import DataFrame
from scipy.stats import linregress

from ..tables import XYTable
from ..graphing import LineGraph


class SimpleLinearRegression:

    _p = "slope", "intercept", "rvalue", "pvalue", "stderr"

    def __init__(self, dt: XYTable) -> None:
        self.dt = dt
        self._params = None

    def solve(self) -> None:
        n = self.dt.n_y_replicates
        x = tile(self.dt.mean.values.T[0], n)
        fit_data = []
        for group in self.dt.groupnames:
            y = self.dt.data[group].values.T.ravel()
            result = linregress(x, y)
            cur_fit = dict(zip(self._p, result))
            cur_fit["group"] = group
            fit_data.append(cur_fit)
        self._params = DataFrame(fit_data).set_index("group")
        return self.params

    def graph(self, n_points: int = 1000, **graph_kw) -> LineGraph:
        # Set default marker style for LinearRegression
        if not "marker" in graph_kw:
            graph_kw["marker"] = "o"
        graph = LineGraph(self.dt, **graph_kw)
        graph.plot()
        x_data = self.dt.data[self.dt.x_name].values
        x_min, x_max = min(x_data), max(x_data)
        x_buffer = (x_max - x_min) * 0.1
        x_min, x_max = x_min - x_buffer, x_max + x_buffer
        x = linspace(x_min, x_max, n_points)
        params = self.params[["slope", "intercept"]]
        for group, *params in params.itertuples():
            y = self._eq(x, *params)
            graph.axes.plot(x, y, label=group, zorder=-1)
        return graph

    def interpolate_extrapolate(self, x: ndarray[float]) -> ndarray[float]:
        return self._eq(array(x), self._m, self._c)

    def inv_interpolate_extrapolate(self, y: ndarray[float]) -> ndarray[float]:
        return self._inv(array(y), self._m, self._c)

    @property
    def params(self) -> DataFrame:
        if self._params is None:
            raise AttributeError(
                "regression analysis yet to be performed, call solve()"
            )
        return self._params

    @property
    def _m(self) -> ndarray:
        return self.params["slope"].values.T

    @property
    def _c(self) -> ndarray:
        return self.params["intercept"].values.T

    def _eq(self, x: float, m: float, c: float) -> float:
        return m*x+c

    def _inv(self, y: float, m: float, c: float) -> float:
        return (y-c)/m
