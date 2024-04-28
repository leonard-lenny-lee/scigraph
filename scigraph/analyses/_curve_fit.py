from __future__ import annotations

__all__ = ["CurveFit"]

from abc import ABC, abstractmethod
from typing import Literal, NamedTuple, Optional, override, TYPE_CHECKING
import inspect

from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Index, MultiIndex
from scipy.optimize import curve_fit
import scipy.stats

from .abc import GraphableAnalysis
from scigraph._log import LOG
from scigraph._options import ConstraintType

if TYPE_CHECKING:
    from scigraph.datatables import XYTable
    from scigraph.graphs import XYGraph


class CurveFit(GraphableAnalysis, ABC):

    def __init__(
        self,
        table: XYTable,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        confidence_level: float = 0.95,
    ) -> None:
        self._table = table

        # Fit attributes
        self.p0: NDArray | None = None
        self._upper_bounds = np.full(self._n_params, +np.inf)
        self._lower_bounds = np.full(self._n_params, -np.inf)
        self._bounds = self._lower_bounds, self._upper_bounds
        self._init_include_list(include, exclude)
        self._confidence_level = confidence_level

    @property
    @override
    def table(self) -> XYTable:
        return self._table

    @override
    def analyze(self) -> DataFrame:
        self.fit()
        return self.result

    @property
    @override
    def result(self) -> DataFrame:
        if hasattr(self, "_pretty_result"):
            return self._pretty_result

        data = []
        for r in self._result.values():
            fit_metrics = np.array([r.dof, r.r2, r.rss, r.sy_x, r.n])
            d = np.hstack([r.popt, r.ci, fit_metrics])
            data.append(d)

        data = np.vstack(data)
        index = [(category, param)
                 for category in ("Best Fit Params",
                                 f"CI {self._confidence_level:.0%}")
                 for param in self._params()]
        index.extend([("Goodness of Fit", stat)
                      for stat in ["dof", "r2", "rss", "sy_x", "n"]])
        index = MultiIndex.from_tuples(index)
        columns = Index(self.table.dataset_ids)
        self._pretty_result = DataFrame(data.T, index, columns)

        return self._pretty_result

    def add_constraint(
        self,
        param: str,
        ty: Literal['equal', "greater", "less"],
        value: float,
        *,
        epsilon: float = 0.01,
    ) -> None:
        try:
            i = self._params().index(param)
        except ValueError as e:
            raise ValueError(
                f"{param} is not a valid parameter. Valid options: "
                f"{", ".join(self._params())}"
            ) from e

        constraint_t = ConstraintType.from_str(ty)
        match constraint_t:
            case ConstraintType.EQUAL:
                err = abs(value) * epsilon / 2
                self._upper_bounds[i] = value + err
                self._lower_bounds[i] = value - err
            case ConstraintType.GREATER:
                self._lower_bounds[i] = value
            case ConstraintType.LESS:
                self._upper_bounds[i] = value

    def fit(self) -> dict[str, CurveFitResult]:
        if self.table.n_x_replicates > 1:
            LOG.warn("Curve fit does not currently support multiple X values. "
                     "X values have been averaged.")

        x_ = self.table.x_values.mean(axis=1)

        assert x_.ndim == 1 and len(x_) == self.table.nrows
        x_ = np.tile(x_, self.table.n_y_replicates)

        result: dict[str, CurveFitResult] = {}

        for id, data in self.table.datasets_itertuples():
            if id not in self._include:
                continue

            y = data.y.T.flatten()
            nan_mask = ~np.isnan(y)
            x, y = x_[nan_mask], y[nan_mask]
            n = len(y)
            dof = n - self._n_params
            assert len(x) == len(y)

            try:
                popt, pcov = curve_fit(self._f, x, y, p0=self.p0,
                                       bounds=self._bounds, nan_policy="omit")
            except RuntimeError as e:
                LOG.warn(f"Curve fit failed for {id}. SciPy error: {e}")
                null = np.full(self._n_params, np.nan)
                r = CurveFitResult(null, null, null, dof, np.nan, np.nan,
                                   np.nan, n, False)
                result[id] = r
                continue
            
            rss = ((y - self._f(x, *popt)) ** 2).sum()
            tss = ((y - y.mean()) ** 2).sum()
            r2 = 1 - (rss / tss)
            sy_x = (rss / dof) ** 0.5

            se = np.sqrt(np.diag(pcov))
            t = scipy.stats.t.ppf((1 + self._confidence_level) / 2, df=dof)
            ci = t * se

            result[id] = CurveFitResult(popt, pcov, ci, dof, r2, rss, sy_x, n,  # type: ignore
                                        True)

        self._result = result
        return self._result

    def interpolate(self, x: NDArray, dataset: str) -> NDArray:
        self._access_check(dataset)
        popt = self._result[dataset].popt
        return self._f(x, *popt)

    @override
    def draw(
        self,
        graph: XYGraph,
        ax: Axes,
        x_min: int | None = None,
        x_max: int | None = None,
        n_points: int = 1000,
        *args,
        **kwargs,
    ) -> None:
        # Determine x limits
        x = self.table.x_values.flatten()
        if graph.xaxis.scale == "log10":
            x = x[np.nonzero(x)]
        if x_min is None:
            x_min = x.min()
        if x_max is None:
            x_max = x.max()

        match graph.xaxis.scale:
            case "linear":
                xlim = x_min, x_max
                x = np.linspace(*xlim, n_points)  # type: ignore
            case "log10":
                xlim = np.log10(x_min), np.log10(x_max)  # type: ignore
                x = np.logspace(*xlim, n_points)
            case _:
                raise NotImplementedError

        for dataset_id, r in self._result.items():
            if not r.converged:
                continue
            props = graph.plot_properties[dataset_id]
            y = self.interpolate(x, dataset_id)
            line, = ax.plot(x, y, *args, **kwargs, **props.line_kws())
            graph._add_legend_artist(dataset_id, line)

    @staticmethod
    @abstractmethod
    def _f(x: NDArray, *args: float) -> NDArray: ...

    def _params(self) -> tuple[str, ...]:
        return tuple(inspect.signature(self._f).parameters.keys())[1:]

    @property
    def _n_params(self) -> int:
        return len(self._params())

    def _init_include_list(
        self,
        include: list[str] | None,
        exclude: list[str] | None,
    ) -> None:
        if include is not None and exclude is not None:
            raise ValueError("Include and exclude cannot both be specified.")
        
        if include is not None:
            self._include = include
        elif exclude is not None:
            self._include = [dataset for dataset in self._table.dataset_ids
                             if dataset not in exclude]
        else:
            self._include = self._table.dataset_ids

    def _access_check(self, dataset: Optional[str] = None) -> None:
        if not hasattr(self, "_result"):
            raise AttributeError("Curve has not been fitted. Call .fit()")

        if dataset is not None and dataset not in self._result:
            raise ValueError(f"{dataset} is not a valid dataset. Available "
                             f"datasets: {self.table.dataset_ids}")


class CurveFitResult(NamedTuple):
    popt: NDArray[np.float64]
    pcov: NDArray[np.float64]
    ci: NDArray[np.float64]
    dof: int
    r2: float
    rss: float
    sy_x: float
    n: int
    converged: bool


class Constant(CurveFit):

    @override
    @staticmethod
    def _f(x: NDArray, constant: float) -> NDArray:  # type: ignore
        return np.full(x.shape, constant)


class SimpleLinearRegression(CurveFit):

    @override
    @staticmethod
    def _f(x: NDArray, slope: float, intercept: float) -> NDArray:  # type: ignore
        return x * slope + intercept


class OnePhaseDecay(CurveFit):
    
    @override
    @staticmethod
    def _f(x: NDArray, y0: float, plateau: float, k: float) -> NDArray:  # type: ignore
        return (y0 - plateau) * np.exp(-k * x) + plateau


class AgonistResponse4Parameter(CurveFit):

    @override
    @staticmethod
    def _f(  # type: ignore
        x: NDArray,
        hill: float,
        top: float,
        bottom: float,
        ec50: float
    ) -> NDArray:
        return bottom+(x**hill)*(top-bottom)/(x**hill+ec50**hill)


class InhibitorResponse4Parameter(CurveFit):

    @override
    @staticmethod
    def _f(  # type: ignore
        x: NDArray,
        hill: float,
        top: float,
        bottom: float,
        ic50: float
    ) -> NDArray:
        return bottom+(top-bottom)/(1+(ic50/x)**hill)
