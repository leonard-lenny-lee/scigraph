from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, override, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from matplotlib.axes import Axes

from scigraph.graphs.abc import GraphComponent 
from scigraph._options import ColumnGraphDirection, ErrorbarType
import scigraph.analyses._agg as agg

if TYPE_CHECKING:
    from scigraph.graphs import XYGraph, ColumnGraph


class ErrorBars(GraphComponent, ABC):

    @override
    def draw_xy(
        self,
        graph: XYGraph,
        ax: Axes,
        *args,
        **kwargs,
    ) -> None:
        draw_x_errors = graph.table.n_x_replicates > 1
        draw_y_errors = graph.table.n_y_replicates > 1

        if not draw_x_errors and not draw_y_errors:
            return

        ori, err = self._prepare_xy(graph)
        x = ori[graph.table.x_title]
        x_err = err[graph.table.x_title] if draw_x_errors else None

        for id in graph.table.dataset_ids:
            y = ori[id]
            y_err = err[id] if draw_y_errors else None
            ax.errorbar(x, y, xerr=x_err, yerr=y_err,
                        *args, **kwargs, marker="", ls="")

    @override
    def draw_column(
        self,
        graph: ColumnGraph,
        ax: Axes,
        *args,
        **kwargs
    ) -> None:
        x = np.linspace(0, graph.table.ncols - 1, graph.table.ncols)
        yori, yerr = self._prepare_column(graph)
        xerr = None
        if graph._direction is ColumnGraphDirection.HORIZONTAL:
            x, yori, xerr, yerr = yori, x, yerr, xerr
        ax.errorbar(x, yori, xerr=xerr, yerr=yerr,
                    *args, **kwargs, color="k", marker="", ls="")

    @abstractmethod
    def _prepare_xy(self, graph: XYGraph, /) -> tuple[DataFrame, Errors]: ...

    @abstractmethod
    def _prepare_column(self, graph: ColumnGraph, /) -> tuple[NDArray, NDArray]: ...

    @classmethod
    def from_opt(cls, opt: ErrorbarType, **_) -> ErrorBars:
        return _FACTORY_MAP[opt]()


class SDErrorBars(ErrorBars):

    @override
    def _prepare_xy(self, graph: XYGraph) -> tuple[DataFrame, Errors]:
        ori = graph.table._reduce_by_row_dataset_column(agg.Basic.mean)
        err = graph.table._reduce_by_row_dataset_column(agg.Basic.sd)
        return ori, err

    @override
    def _prepare_column(self, graph: ColumnGraph) -> tuple[NDArray, NDArray]:
        yori = graph.table._reduce_by_column(agg.Basic.mean)
        yerr = graph.table._reduce_by_column(agg.Basic.sd)
        return yori, yerr


class SEMErrorBars(ErrorBars):

    @override
    def _prepare_xy(self, graph: XYGraph) -> tuple[DataFrame, Errors]:
        ori = graph.table._reduce_by_row_dataset_column(agg.Basic.mean)
        err = graph.table._reduce_by_row_dataset_column(agg.Basic.sem)
        return ori, err

    @override
    def _prepare_column(self, graph: ColumnGraph) -> tuple[NDArray, NDArray]:
        yori = graph.table._reduce_by_column(agg.Basic.mean)
        yerr = graph.table._reduce_by_column(agg.Basic.sem)
        return yori, yerr


class CI95ErrorBars(ErrorBars):

    @override
    def _prepare_xy(self, graph: XYGraph) -> tuple[DataFrame, Errors]:
        ori = graph.table._reduce_by_row_dataset_column(agg.Basic.mean)
        err = graph.table._reduce_by_row_dataset_column(
            agg.ConfidenceInterval.mean
        )
        return ori, err

    @override
    def _prepare_column(self, graph: ColumnGraph) -> tuple[NDArray, NDArray]:
        yori = graph.table._reduce_by_column(agg.Basic.mean)
        yerr = graph.table._reduce_by_column(
            agg.ConfidenceInterval.mean
        )
        return yori, yerr


class RangeErrorBars(ErrorBars):

    @override
    def _prepare_xy(self, graph: XYGraph) -> tuple[DataFrame, Errors]:
        ori = graph.table._reduce_by_row_dataset_column(agg.Basic.mean)
        lower = ori - graph.table._reduce_by_row_dataset_column(agg.Basic.min)
        upper = graph.table._reduce_by_row_dataset_column(agg.Basic.max) - ori
        err = {col: (lower[col].values, upper[col].values) for col in ori}
        return ori, err

    @override
    def _prepare_column(self, graph: ColumnGraph) -> tuple[NDArray, NDArray]:
        ori = graph.table._reduce_by_column(agg.Basic.mean)
        lower = ori - graph.table._reduce_by_column(agg.Basic.min)
        upper = graph.table._reduce_by_column(agg.Basic.max) - ori
        err = np.vstack((lower, upper))
        return ori, err


class GeometricSDErrorBars(ErrorBars):

    @override
    def _prepare_xy(self, graph: XYGraph) -> tuple[DataFrame, Errors]:
        ori = graph.table._reduce_by_row_dataset_column(
            agg.Advanced.geometric_mean
        )
        err = graph.table._reduce_by_row_dataset_column(
            agg.Advanced.geometric_sd
        )
        return ori, err

    @override
    def _prepare_column(self, graph: ColumnGraph) -> tuple[NDArray, NDArray]:
        yori = graph.table._reduce_by_column(agg.Advanced.geometric_mean)
        yerr = graph.table._reduce_by_column(agg.Advanced.geometric_sd)
        return yori, yerr


_FACTORY_MAP: dict[ErrorbarType, type[ErrorBars]] = {
    ErrorbarType.SD: SDErrorBars,
    ErrorbarType.SEM: SEMErrorBars,
    ErrorbarType.CI95: CI95ErrorBars,
    ErrorbarType.RANGE: RangeErrorBars,
    ErrorbarType.GEOMETRIC_SD: GeometricSDErrorBars,
}

type Errors = DataFrame | Mapping[str, tuple[NDArray, NDArray]]
