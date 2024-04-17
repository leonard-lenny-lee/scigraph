from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, override, Self, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from matplotlib.axes import Axes

from scigraph.graphs.abc import Artist, TypeChecked
from scigraph._options import (GraphType, XYGraphSubtype, ColumnGraphSubtype,
                               ColumnGraphDirection)
import scigraph.analyses._agg as agg

if TYPE_CHECKING:
    from scigraph.graphs import XYGraph, ColumnGraph


class ErrorBars(Artist, TypeChecked, ABC):

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
    def _prepare_xy(self, graph: XYGraph) -> tuple[DataFrame, Errors]: ...

    @abstractmethod
    def _prepare_column(self, graph: ColumnGraph) -> tuple[NDArray, NDArray]: ...

    @classmethod
    def from_str(cls, s: str) -> Self | None:
        if s in _FACTORY_MAP:
            return _FACTORY_MAP[s]()
        return None


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

    @override
    @classmethod
    def _compatible_types(cls) -> set[TypeChecked.Type]:
        return {
            TypeChecked.Type(GraphType.XY, XYGraphSubtype.MEAN),
            TypeChecked.Type(GraphType.COLUMN, ColumnGraphSubtype.MEAN)
        }


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

    @classmethod
    def _compatible_types(cls) -> set[TypeChecked.Type]:
        return {
            TypeChecked.Type(GraphType.XY, XYGraphSubtype.MEAN),
            TypeChecked.Type(GraphType.COLUMN, ColumnGraphSubtype.MEAN)
        }


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

    @override
    @classmethod
    def _compatible_types(cls) -> set[TypeChecked.Type]:
        return {
            TypeChecked.Type(GraphType.XY, XYGraphSubtype.MEAN),
            TypeChecked.Type(GraphType.COLUMN, ColumnGraphSubtype.MEAN),
        }


class RangeErrorBars(ErrorBars):

    @override
    def _prepare_xy(self, graph: XYGraph) -> tuple[DataFrame, Errors]:
        ori = graph.table._reduce_by_row_dataset_column(
            agg.agg(graph._subtype.to_str())
        )
        lower = ori - graph.table._reduce_by_row_dataset_column(agg.Basic.min)
        upper = graph.table._reduce_by_row_dataset_column(agg.Basic.max) - ori
        err = {col: (lower[col].values, upper[col].values) for col in ori}
        return ori, err

    @override
    def _prepare_column(self, graph: ColumnGraph) -> tuple[NDArray, NDArray]:
        ori = graph.table._reduce_by_column(agg.agg(graph._subtype.to_str()))
        lower = ori - graph.table._reduce_by_column(agg.Basic.min)
        upper = graph.table._reduce_by_column(agg.Basic.max) - ori
        err = np.vstack((lower, upper))
        return ori, err

    @override
    @classmethod
    def _compatible_types(cls) -> set[TypeChecked.Type]:
        return {
            TypeChecked.Type(GraphType.XY, XYGraphSubtype.MEAN),
            TypeChecked.Type(GraphType.XY, XYGraphSubtype.MEDIAN),
            TypeChecked.Type(GraphType.COLUMN, ColumnGraphSubtype.MEAN),
            TypeChecked.Type(GraphType.COLUMN, ColumnGraphSubtype.MEDIAN),
        }


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

    @override
    @classmethod
    def _compatible_types(cls) -> set[TypeChecked.Type]:
        return {
            TypeChecked.Type(GraphType.XY, XYGraphSubtype.GEOMETRIC_MEAN),
            TypeChecked.Type(
                GraphType.COLUMN, ColumnGraphSubtype.GEOMETRIC_MEAN
            ),
        }


_FACTORY_MAP = {
    "sd": SDErrorBars,
    "sem": SEMErrorBars,
    "ci95": CI95ErrorBars,
    "range": RangeErrorBars,
    "geometric sd": GeometricSDErrorBars,
}

type Errors = DataFrame | Mapping[str, tuple[NDArray, NDArray]]
