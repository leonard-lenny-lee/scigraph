from abc import abstractmethod
from dataclasses import dataclass
import logging
from typing import override

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.optimize import curve_fit

from .abc import XYAnalysis, Plottable
from scigraph.datatables.xy import XYTable
from scigraph.graphs.xy import XYPointsGraph, ScaleOpt


class CurveFit(Plottable, XYAnalysis):

    def __init__(
        self,
        table: XYTable,
    ) -> None:
        super().__init__(table)
        self.plot_params = _PlotParams()
        self.p0 = None
        self.bounds = None

    def analyze(self) -> pd.DataFrame:
        return self.fit()

    def fit(self) -> None:
        popts = {}
        pcovs = {}
        df = self.table.as_df()
        x = np.tile(df.values[:, 0].flatten(), self.table.n_replicates)
        for group in self.table.ygroups:
            y = df[group].values.T.flatten()
            assert len(x) == len(y)
            try:
                popt, pcov = curve_fit(self._f, x, y)
            except RuntimeError as e:
                logging.warn(e)
                popt, pcov = None, None
            popts[group] = popt
            pcovs[group] = pcov
        self.popt = pd.DataFrame(popts, index=self._params())
        self.pcov = pcovs

    def predict(self, x: NDArray, group: str) -> NDArray:
        popt = self.popt[group]
        return self._f(x, *popt)

    @override
    def plot(
        self,
        ax: plt.Axes,
        graph: XYPointsGraph,
    ) -> None:
        ax.set_prop_cycle(None)  # Reset
        x = self.table.xvalues[:, 0]
        match graph.xaxis_opts.scale:
            case ScaleOpt.LINEAR:
                xlim = x.min(), x.max()
                x = np.linspace(*xlim, self.plot_params.n_points)
            case ScaleOpt.LOG10:
                x = x[np.nonzero(x)]
                xlim = np.log10(x.min()), np.log10(x.max())
                x = np.logspace(*xlim, self.plot_params.n_points)
        for group in self.table.ygroups:
            if self.popt[group] is None:
                ax.plot([], [])
                continue
            y = self.predict(x, group)
            handle, = ax.plot(x, y, marker="")
            graph._handles[group].append(handle)

    @staticmethod
    @abstractmethod
    def _f(x: NDArray) -> NDArray: ...

    @staticmethod
    @abstractmethod
    def _params() -> tuple[str]: ...


class SimpleLinearRegression(CurveFit):

    @override
    @staticmethod
    def _f(x, slope, intercept):
        return x * slope + intercept

    @override
    @staticmethod
    def _params() -> tuple[str]:
        return "Slope", "Intercept"


class AgonistResponse4Parameter(CurveFit):

    @override
    @staticmethod
    def _f(x, hill, top, bottom, ec50) -> NDArray:
        return bottom+(x**hill)*(top-bottom)/(x**hill+ec50**hill)

    @override
    @staticmethod
    def _params() -> tuple[str]:
        return "Hill Slope", "Top", "Bottom", "EC50"


class InhibitorResponse4Parameter(CurveFit):

    @override
    @staticmethod
    def _f(x, hill, top, bottom, ic50) -> NDArray:
        return bottom+(top-bottom)/(1+(ic50/x)**hill)

    @override
    @staticmethod
    def _params() -> tuple[str]:
        return "Hill Slope", "Top", "Bottom", "IC50"


@dataclass
class _PlotParams:
    n_points: int = 1000
