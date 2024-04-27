from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, override, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from matplotlib.axes import Axes

from scigraph.graphs.abc import GraphComponent 
from scigraph._options import ErrorbarType
import scigraph.analyses._stats as sgstats

if TYPE_CHECKING:
    from scigraph.graphs import XYGraph, ColumnGraph, GroupedGraph


class ErrorBars(GraphComponent, ABC):

    @override
    def draw_xy(self, graph: XYGraph, ax: Axes) -> None:
        draw_x_errors = graph.table.n_x_replicates > 1
        draw_y_errors = graph.table.n_y_replicates > 1

        if not draw_x_errors and not draw_y_errors:
            return

        ori, err = self._prepare_xy(graph)
        x = ori[graph.table.x_title]
        x_err = err[graph.table.x_title] if draw_x_errors else None

        for id in graph.table.dataset_ids:
            (props := graph.plot_properties[id].errorbar_kw()).update(**self.kw)
            y = ori[id]
            y_err = err[id] if draw_y_errors else None
            ax.errorbar(x, y, xerr=x_err, yerr=y_err, 
                        **self.kw, **props)

    @override
    def draw_column(self, graph: ColumnGraph, ax: Axes) -> None:
        x = np.linspace(0, graph.table.ncols - 1, graph.table.ncols)
        yori, yerr = self._prepare_column(graph)
        xerr = [None for _ in range(graph.table.ncols)]
        if not graph._is_vertical:
            x, yori, xerr, yerr = yori, x, yerr, xerr

        for i, id in enumerate(graph.table.dataset_ids):
            (props := graph.plot_properties[id].errorbar_kw()).update(**self.kw)
            ax.errorbar(x[i], yori[i], xerr=xerr[i], yerr=yerr[i],
                        **self.kw, **props)

    @override
    def draw_grouped(self, graph: GroupedGraph, ax: Axes) -> None:
        if not graph.table._n_replicates > 1:
            return

        x = graph._x()
        yori, yerr = self._prepare_grouped(graph)
        xerr = None

        for i, id in enumerate(graph.table.dataset_ids):
            props = graph.plot_properties[id].errorbar_kw()
            props.update(**self.kw)
            x_, y_, xerr_, yerr_ = x[i], yori[id], xerr, yerr[id]
            if not graph._is_vertical:
                x_, y_, xerr_, yerr_= y_, x_, yerr_, xerr_
            ax.errorbar(x_, y_, xerr=xerr_, yerr=yerr_,
                        **self.kw, **props)

    @abstractmethod
    def _prepare_xy(self, graph: XYGraph, /) -> tuple[DataFrame, Errors]: ...

    @abstractmethod
    def _prepare_column(self, graph: ColumnGraph, /) -> tuple[NDArray, NDArray]: ...

    @abstractmethod
    def _prepare_grouped(self, graph: GroupedGraph, /) -> tuple[DataFrame, Errors]: ...

    @classmethod
    def from_opt(cls, opt: ErrorbarType, kw: dict[str, Any], **_) -> ErrorBars:
        return _FACTORY_MAP[opt](kw)


class SDErrorBars(ErrorBars):

    @override
    def _prepare_xy(self, graph: XYGraph) -> tuple[DataFrame, Errors]:
        ori = graph.table._row_statistics_by_dataset(sgstats.Basic.mean)
        err = graph.table._row_statistics_by_dataset(sgstats.Basic.sd)
        return ori, err

    @override
    def _prepare_column(self, graph: ColumnGraph) -> tuple[NDArray, NDArray]:
        yori = graph.table._reduce_by_column(sgstats.Basic.mean)
        yerr = graph.table._reduce_by_column(sgstats.Basic.sd)
        return yori, yerr

    @override
    def _prepare_grouped(self, graph: GroupedGraph) -> tuple[DataFrame, Errors]:
        ori = graph.table._row_statistics_by_dataset(sgstats.Basic.mean)
        err = graph.table._row_statistics_by_dataset(sgstats.Basic.sd)
        return ori, err


class SEMErrorBars(ErrorBars):

    @override
    def _prepare_xy(self, graph: XYGraph) -> tuple[DataFrame, Errors]:
        ori = graph.table._row_statistics_by_dataset(sgstats.Basic.mean)
        err = graph.table._row_statistics_by_dataset(sgstats.Basic.sem)
        return ori, err

    @override
    def _prepare_column(self, graph: ColumnGraph) -> tuple[NDArray, NDArray]:
        yori = graph.table._reduce_by_column(sgstats.Basic.mean)
        yerr = graph.table._reduce_by_column(sgstats.Basic.sem)
        return yori, yerr

    @override
    def _prepare_grouped(self, graph: GroupedGraph) -> tuple[DataFrame, Errors]:
        ori = graph.table._row_statistics_by_dataset(sgstats.Basic.mean)
        err = graph.table._row_statistics_by_dataset(sgstats.Basic.sem)
        return ori, err


class CI95ErrorBars(ErrorBars):

    @override
    def _prepare_xy(self, graph: XYGraph) -> tuple[DataFrame, Errors]:
        ori = graph.table._row_statistics_by_dataset(sgstats.Basic.mean)
        err = graph.table._row_statistics_by_dataset(
            sgstats.ConfidenceInterval.mean
        )
        return ori, err

    @override
    def _prepare_column(self, graph: ColumnGraph) -> tuple[NDArray, NDArray]:
        yori = graph.table._reduce_by_column(sgstats.Basic.mean)
        yerr = graph.table._reduce_by_column(
            sgstats.ConfidenceInterval.mean
        )
        return yori, yerr

    @override
    def _prepare_grouped(self, graph: GroupedGraph) -> tuple[DataFrame, Errors]:
        ori = graph.table._row_statistics_by_dataset(sgstats.Basic.mean)
        err = graph.table._row_statistics_by_dataset(
            sgstats.ConfidenceInterval.mean
        )
        return ori, err



class RangeErrorBars(ErrorBars):

    @override
    def _prepare_xy(self, graph: XYGraph) -> tuple[DataFrame, Errors]:
        ori = graph.table._row_statistics_by_dataset(sgstats.Basic.mean)
        lower = ori - graph.table._row_statistics_by_dataset(sgstats.Basic.min)
        upper = graph.table._row_statistics_by_dataset(sgstats.Basic.max) - ori
        err = {col: (lower[col].values, upper[col].values) for col in ori}
        return ori, err

    @override
    def _prepare_column(self, graph: ColumnGraph) -> tuple[NDArray, NDArray]:
        ori = graph.table._reduce_by_column(sgstats.Basic.mean)
        lower = ori - graph.table._reduce_by_column(sgstats.Basic.min)
        upper = graph.table._reduce_by_column(sgstats.Basic.max) - ori
        err = np.vstack((lower, upper))
        return ori, err

    @override
    def _prepare_grouped(self, graph: GroupedGraph) -> tuple[DataFrame, Errors]:
        ori = graph.table._row_statistics_by_dataset(sgstats.Basic.mean)
        lower = ori - graph.table._row_statistics_by_dataset(sgstats.Basic.min)
        upper = graph.table._row_statistics_by_dataset(sgstats.Basic.max) - ori
        err = {col: (lower[col].values, upper[col].values) for col in ori}
        return ori, err


class GeometricSDErrorBars(ErrorBars):

    @override
    def _prepare_xy(self, graph: XYGraph) -> tuple[DataFrame, Errors]:
        ori = graph.table._row_statistics_by_dataset(
            sgstats.Advanced.geometric_mean
        )
        err = graph.table._row_statistics_by_dataset(
            sgstats.Advanced.geometric_sd
        )
        return ori, err

    @override
    def _prepare_column(self, graph: ColumnGraph) -> tuple[NDArray, NDArray]:
        yori = graph.table._reduce_by_column(sgstats.Advanced.geometric_mean)
        yerr = graph.table._reduce_by_column(sgstats.Advanced.geometric_sd)
        return yori, yerr

    @override
    def _prepare_grouped(self, graph: GroupedGraph) -> tuple[DataFrame, Errors]:
        ori = graph.table._row_statistics_by_dataset(
            sgstats.Advanced.geometric_mean
        )
        err = graph.table._row_statistics_by_dataset(
            sgstats.Advanced.geometric_sd
        )
        return ori, err


_FACTORY_MAP: dict[ErrorbarType, type[ErrorBars]] = {
    ErrorbarType.SD: SDErrorBars,
    ErrorbarType.SEM: SEMErrorBars,
    ErrorbarType.CI95: CI95ErrorBars,
    ErrorbarType.RANGE: RangeErrorBars,
    ErrorbarType.GEOMETRIC_SD: GeometricSDErrorBars,
}

type Errors = DataFrame | Mapping[str, tuple[NDArray, NDArray]]
