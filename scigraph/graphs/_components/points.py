from __future__ import annotations

from abc import abstractmethod
from typing import override, TYPE_CHECKING
import logging

from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from ..abc import TypeChecked, Artist
from scigraph._options import (GraphType, XYGraphSubtype, ColumnGraphSubtype,
                               ColumnGraphDirection)
import scigraph.analyses._agg as agg

if TYPE_CHECKING:
    from scigraph.graphs import XYGraph, ColumnGraph


class Points(Artist, TypeChecked):

    @override
    def draw_xy(
        self,
        graph: XYGraph,
        ax: Axes,
        *args,
        **kwargs,
    ) -> None:
        agg = self._prepare_xy(graph)
        x = agg[graph.table.x_title]

        for id in graph.table.dataset_ids:
            props = graph.plot_properties[id]
            y = agg[id]
            artist, = ax.plot(x, y, *args, **kwargs,
                              marker=props.marker, color=props.color, ls="")
            graph._add_legend_artist(id, artist)

    @override
    def draw_column(
        self,
        graph: ColumnGraph,
        ax: Axes,
        *args,
        **kwargs
    ) -> None:
        x = np.linspace(0, graph.table.ncols - 1, graph.table.ncols)
        y = self._prepare_column(graph)
        if graph._direction is ColumnGraphDirection.HORIZONTAL:
            x, y = y, x
        ax.plot(x, y, *args, **kwargs, marker="o", color="k", ls="")

    @abstractmethod
    def _prepare_xy(self, graph: XYGraph) -> DataFrame: ...

    @abstractmethod
    def _prepare_column(self, graph: ColumnGraph) -> NDArray: ...

    @classmethod
    def from_str(cls, s: str) -> Points | None:
        if s in _FACTORY_MAP:
            return _FACTORY_MAP[s]()
        return None


class MeanPoints(Points):

    @override
    def _prepare_xy(self, graph: XYGraph) -> DataFrame:
        return graph.table._reduce_by_row_dataset_column(agg.Basic.mean)

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(agg.Basic.mean)

    @override
    @classmethod
    def _compatible_types(cls) -> set[TypeChecked.Type]:
        return {
            TypeChecked.Type(GraphType.XY, XYGraphSubtype.MEAN),
            TypeChecked.Type(GraphType.COLUMN, ColumnGraphSubtype.MEAN),
        }


class GeometricMeanPoints(Points):

    @override
    def _prepare_xy(self, graph: XYGraph) -> DataFrame:
        return graph.table._reduce_by_row_dataset_column(
            agg.Advanced.geometric_mean
        )

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(agg.Advanced.geometric_mean)

    @override
    @classmethod
    def _compatible_types(cls) -> set[TypeChecked.Type]:
        return {
            TypeChecked.Type(GraphType.XY, XYGraphSubtype.GEOMETRIC_MEAN),
            TypeChecked.Type(GraphType.COLUMN, ColumnGraphSubtype.GEOMETRIC_MEAN),
        }

class MedianPoints(Points):

    @override
    def _prepare_xy(self, graph: XYGraph) -> DataFrame:
        return graph.table._reduce_by_row_dataset_column(agg.Basic.median)

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(agg.Basic.median)

    @override
    @classmethod
    def _compatible_types(cls) -> set[TypeChecked.Type]:
        return {
            TypeChecked.Type(GraphType.XY, XYGraphSubtype.MEDIAN),
            TypeChecked.Type(GraphType.COLUMN, ColumnGraphSubtype.MEDIAN),
        }


class IndividualPoints(Points):

    @override
    def draw_xy(
        self,
        graph: XYGraph,
        ax: Axes,
        *args,
        **kwargs,
    ) -> None:
        # No aggregation
        x = graph.table.x_values
        if graph.table.n_x_replicates > 1:
            x = x.mean(axis=1)
            logging.warn(
                "Multiple X values are incompatible with individual points "
                "and have been averaged.")
        # Match y series dimensions
        x = np.tile(x.flatten(), graph.table.n_y_replicates)

        for id, dataset in graph.table.datasets_itertuples():
            props = graph.plot_properties[id]
            y = dataset.y.T.flatten()
            assert len(x) == len(y)
            artist, = ax.plot(x, y, *args, **kwargs,
                              marker=props.marker, color=props.color, ls="")
            graph._add_legend_artist(id, artist)

    @override
    def draw_column(
        self,
        graph: ColumnGraph,
        ax: Axes,
        *args,
        **kwargs
    ) -> None:
        x = np.linspace(0, graph.table.ncols - 1, graph.table.ncols)
        x = np.repeat(x, graph.table.nrows)
        y = graph.table.values.flatten('F')
        if graph._direction is ColumnGraphDirection.HORIZONTAL:
            x, y = y, x
        ax.plot(x, y, *args, **kwargs, marker="o", color="k", ls="")

    @override
    def _prepare_xy(self, graph: XYGraph) -> DataFrame:
        raise NotImplementedError

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        raise NotImplementedError

    @override
    @classmethod
    def _compatible_types(cls) -> set[TypeChecked.Type]:
        return {
            TypeChecked.Type(GraphType.XY, XYGraphSubtype.INDIVIDUAL),
            TypeChecked.Type(GraphType.COLUMN, ColumnGraphSubtype.INDIVIDUAL),
        }


class SwarmPoints(Points): ...  # TODO


_FACTORY_MAP: dict[str, type[Points]] = {
    "mean": MeanPoints,
    "geometric mean": GeometricMeanPoints,
    "median": MedianPoints,
    "individual": IndividualPoints,
    "swarm": SwarmPoints,
}
