from abc import ABC, abstractmethod
import logging
from typing import Literal, override

from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Index
from scipy.optimize import curve_fit

from .abc import GraphableAnalysis
from scigraph.datatables import XYTable
from scigraph.graphs import XYGraph


class CurveFit(GraphableAnalysis[XYTable, XYGraph], ABC):

    def __init__(
        self,
        table: XYTable,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        self._table = table

        # Fit attributes
        self.p0: NDArray | None = None
        self._upper_bounds = np.full(len(self._params()), +np.nan)
        self._lower_bounds = np.full(len(self._params()), -np.nan)
        self._bounds = self._lower_bounds, self._upper_bounds
        self._init_include_list(include, exclude)

        # Results
        self._popt: dict[str, tuple[float, ...]] = {}

    @property
    @override
    def table(self) -> XYTable:
        return self._table

    @property
    def popt(self) -> DataFrame:
        return DataFrame(self._popt, index=Index(self._params()))

    @override
    def analyze(self) -> DataFrame:
        return self.fit()

    def add_constraint(
        self,
        param: str,
        ty: Literal['=', ">", "<"],
        value: float,
        *,
        alpha: float = 0.01,
    ) -> None:
        try:
            i = self._params().index(param)
        except ValueError as e:
            raise ValueError(
                f"{param} is not a valid parameter. Valid options: " +
                f"{", ".join(self._params())}"
            ) from e
        match ty:
            case "=":
                err = abs(value) * alpha / 2
                self._upper_bounds[i] = value + err
                self._lower_bounds[i] = value - err
            case ">":
                self._lower_bounds[i] = value
            case "<":
                self._upper_bounds[i] = value
            case _:
                raise ValueError("Invalid constraint type argument.")

    def fit(self) -> DataFrame:
        popts = {}
        pcovs = {}

        x = self.table.x_values.mean(axis=1)
        assert x.ndim == 1 and len(x) == self.table.nrows
        x = np.tile(x, self.table.n_y_replicates)

        for id, data in self.table.datasets_itertuples():
            if id not in self._include:
                continue

            y = data.y.T.flatten()
            assert len(x) == len(y)
            # Remove NaN values
            mask = ~(np.isnan(y) | np.isnan(x))
            try:
                popt, pcov = curve_fit(self._f, x[mask], y[mask],
                                       p0=self.p0, bounds=self._bounds)
            except RuntimeError as e:
                logging.warn(e)
                popt, pcov = None, None

            popts[id] = popt
            pcovs[id] = pcov

        self._popt = popts
        self._pcov = pcovs
        return self.popt

    def predict(self, x: NDArray, dataset: str) -> NDArray:
        popt = self._popt[dataset]
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

        for dataset_id, popt in self._popt.items():
            props = graph.plot_properties[dataset_id]
            if popt is None:
                continue
            y = self.predict(x, dataset_id)
            line, = ax.plot(x, y, *args, **kwargs,
                            c=props.color, marker="", ls=props.linestyle)
            graph._add_legend_artist(dataset_id, line)

    @staticmethod
    @abstractmethod
    def _f(x: NDArray, *args: float) -> NDArray: ...

    @staticmethod
    @abstractmethod
    def _params() -> tuple[str, ...]: ...

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



class Constant(CurveFit):

    @override
    @staticmethod
    def _f(x: NDArray, constant: float) -> NDArray:  # type: ignore
        return np.full(x.shape, constant)

    @override
    @staticmethod
    def _params() -> tuple[str, ...]:
        return "Constant",


class SimpleLinearRegression(CurveFit):

    @override
    @staticmethod
    def _f(x: NDArray, slope: float, intercept: float) -> NDArray:  # type: ignore
        return x * slope + intercept

    @override
    @staticmethod
    def _params() -> tuple[str, ...]:
        return "Slope", "Intercept"


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

    @override
    @staticmethod
    def _params() -> tuple[str, ...]:
        return "Hill Slope", "Top", "Bottom", "EC50"


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

    @override
    @staticmethod
    def _params() -> tuple[str, ...]:
        return "Hill Slope", "Top", "Bottom", "IC50"
