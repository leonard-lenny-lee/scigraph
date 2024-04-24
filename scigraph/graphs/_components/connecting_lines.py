"""
Artists that connect average of replicates together, or individual replicates
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Never, Self, override, TYPE_CHECKING

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from scigraph.graphs.abc import GraphComponent
from scigraph._options import ColumnGraphDirection, ConnectingLineType
import scigraph.analyses._stats as sgstats

if TYPE_CHECKING:
    from scigraph.graphs import XYGraph, ColumnGraph


class ConnectingLine(GraphComponent, ABC):

    def __init__(self, kw: dict[str, Any], join_nan: bool) -> None:
        super().__init__(kw)
        self.join_nan = join_nan

    @override
    def draw_xy(self, graph: XYGraph, ax: Axes) -> None:
        agg = self._prepare_xy(graph)

        for id in graph.table.dataset_ids:
            props = graph.plot_properties[id].line_kws()
            props.update(**self.kw)
            x = np.array(agg[graph.table.x_title].values).flatten()
            y = np.array(agg[id].values).flatten()
            if self.join_nan:
                # Mask NaN values so there is a continuous joined line
                x, y = self._mask_nan(x, y)
            artist, = ax.plot(x, y, **props)
            graph._add_legend_artist(id, artist)

    @override
    def draw_column(self, graph: ColumnGraph,  ax: Axes) -> None:
        x = np.linspace(0, graph.table.ncols - 1, graph.table.ncols)
        y = self._prepare_column(graph)
        if graph._direction is ColumnGraphDirection.HORIZONTAL:
            x, y = y, x

        # It doesn't make sense for connecting lines between groups to have
        # different linestyles so use the first one provided
        line_kws = (
            graph
            .plot_properties[graph.table.dataset_ids[0]]
            .line_kws()
        )
        line_kws.update(**self.kw)
        ax.plot(x, y, **line_kws)

    @abstractmethod
    def _prepare_xy(self, graph: XYGraph, /) -> DataFrame: ...

    @abstractmethod
    def _prepare_column(self, graph: ColumnGraph, /) -> NDArray: ...

    def _mask_nan(self, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        assert x.shape == y.shape
        assert x.ndim == 1
        assert y.ndim == 1

        stacked_array = np.vstack((x, y))
        mask = ~np.any(np.isnan(stacked_array), axis=0)
        masked_array = stacked_array[:, mask]
        return masked_array[0], masked_array[1]

    @classmethod
    def from_opt(
        cls,
        opt: ConnectingLineType,
        kw: dict[str, Any],
        **kwargs,
    ) -> Self:
        return _FACTORY_MAP[opt](kw, **kwargs)


class MeanConnectingLine(ConnectingLine):

    @override
    def _prepare_xy(self, graph: XYGraph) -> DataFrame:
        return graph.table.row_statistics_by_dataset(sgstats.Basic.mean)

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(sgstats.Basic.mean)


class GeometricMeanConnectingLine(ConnectingLine):

    @override
    def _prepare_xy(self, graph: XYGraph) -> DataFrame:
        return graph.table.row_statistics_by_dataset(
            sgstats.Advanced.geometric_mean
        )

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(sgstats.Advanced.geometric_mean)


class MedianConnectingLine(ConnectingLine):

    @override
    def _prepare_xy(self, graph: XYGraph) -> DataFrame:
        return graph.table.row_statistics_by_dataset(sgstats.Basic.median)

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(sgstats.Basic.median)


class IndividualConnectingLine(ConnectingLine):

    @override
    def draw_xy(self, graph: XYGraph, ax: Axes) -> None:
        x = graph.table.x_values.mean(axis=1).flatten()

        for id, data in graph.table.datasets_itertuples():
            props = graph.plot_properties[id].line_kws()
            props.update(**self.kw)
            for y in data.y.T:
                assert x.shape == y.shape
                if self.join_nan:
                    x_, y = self._mask_nan(x, y)
                else:
                    x_ = x
                ax.plot(x_, y, **props)
            artist = Line2D([], [], **props)
            graph._add_legend_artist(id, artist)

    @override
    def draw_column(self, graph: ColumnGraph,  ax: Axes) -> None:
        x = np.linspace(0, graph.table.ncols - 1, graph.table.ncols)
        line_kws = (
            graph
            .plot_properties[graph.table.dataset_ids[0]]
            .line_kws()
        )
        line_kws.update(**self.kw)
        for y in graph.table.values:
            x_ = x
            if graph._direction is ColumnGraphDirection.HORIZONTAL:
                x_, y = y, x_
            ax.plot(x_, y, **line_kws)

    @override
    def _prepare_xy(self, _) -> Never:
        raise NotImplementedError

    @override
    def _prepare_column(self, _) -> Never:
        raise NotImplementedError


_FACTORY_MAP = {
    ConnectingLineType.MEAN: MeanConnectingLine,
    ConnectingLineType.GEOMETRIC_MEAN: GeometricMeanConnectingLine,
    ConnectingLineType.MEDIAN: MedianConnectingLine,
    ConnectingLineType.INDIVIDUAL: IndividualConnectingLine,
}
