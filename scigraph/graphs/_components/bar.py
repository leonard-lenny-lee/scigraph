from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Never, Self, Any, override, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from scigraph.graphs.abc import GraphComponent
from scigraph._options import BarType
import scigraph.analyses._stats as sgstats

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from pandas import DataFrame

    from scigraph.graphs import ColumnGraph, GroupedGraph


class Bars(GraphComponent, ABC):

    def __init__(self, kw: dict[str, Any], line_only: bool = False) -> None:
        super().__init__(kw)
        self.line_only = line_only

    @override
    def draw_xy(self, *args, **kwargs) -> Never:
        raise NotImplementedError

    @override
    def draw_column(self, graph: ColumnGraph, ax: Axes) -> None:
        x = np.linspace(0, graph.table.ncols - 1, graph.table.ncols)
        y = self._prepare_column(graph)

        for i, id in enumerate(graph.table.dataset_ids):
            if not self.line_only:  # Draw full bars
                props = graph.plot_properties[id].bar_kw(graph._is_vertical)
                props.update(**self.kw)
                if graph._is_vertical:
                    ax.bar(x[i], y[i], **props)
                else:  # Horizontal
                    ax.barh(x[i], y[i], **props)
            else:  # Draw top line of bar only
                props = graph.plot_properties[id]
                barl_kw = props.barl_kw()
                barl_kw.update(**self.kw)
                if props.barwidth is None:
                    props.barwidth = 0.5
                x_delta = props.barwidth / 2
                x_ = x[i] - x_delta, x[i] + x_delta
                if graph._is_vertical:
                    ax.hlines(y[i], *x_, **barl_kw)
                else:  # Horizontal
                    ax.vlines(y[i], *x_, **barl_kw)

    @override
    def draw_grouped(self, graph: GroupedGraph, ax: Axes) -> None:
        x = graph._x()
        y = self._prepare_grouped(graph).values.T
        width_adjustment_factor = 1 / (1 + graph.table._n_datasets)

        for x_, y_, id in zip(x, y, graph.table.dataset_ids):
            if not self.line_only:  # Draw full bars
                props = graph.plot_properties[id].bar_kw(graph._is_vertical)
                props.update(**self.kw)
                if graph._is_vertical:
                    props["width"] *= width_adjustment_factor
                    ax.bar(x_, y_, **props)
                else:  # Horizontal
                    props["height"] *= width_adjustment_factor
                    ax.barh(x_, y_, **props)
            else:  # Draw top line of bar only
                props = graph.plot_properties[id]
                barl_kw = props.barl_kw()
                barl_kw.update(**self.kw)
                if props.barwidth is None:
                    props.barwidth = 0.5
                x_delta = props.barwidth * width_adjustment_factor / 2
                x_ = x_ - x_delta, x_ + x_delta
                if graph._is_vertical:
                    ax.hlines(y_, *x_, **barl_kw)
                else:  # Horizontal
                    ax.vlines(y_, *x_, **barl_kw)

    @abstractmethod
    def _prepare_column(self, graph: ColumnGraph) -> NDArray: ...

    @abstractmethod
    def _prepare_grouped(self, graph: GroupedGraph) -> DataFrame: ...

    @override
    @classmethod
    def from_opt(cls, opt: BarType, kw, **kwargs) -> Self:
        return _FACTORY_MAP[opt](kw, **kwargs)


class MeanBars(Bars):

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(sgstats.Basic.mean)

    @override
    def _prepare_grouped(self, graph: GroupedGraph) -> DataFrame:
        return graph.table._row_statistics_by_dataset(sgstats.Basic.mean)


class GeometricMeanBars(Bars):

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(sgstats.Advanced.geometric_mean)

    @override
    def _prepare_grouped(self, graph: GroupedGraph) -> DataFrame:
        return graph.table._row_statistics_by_dataset(sgstats.Advanced.geometric_mean)


class MedianBars(Bars):

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(sgstats.Basic.median)

    @override
    def _prepare_grouped(self, graph: GroupedGraph) -> DataFrame:
        return graph.table._row_statistics_by_dataset(sgstats.Basic.median)


_FACTORY_MAP = {
    BarType.MEAN: MeanBars,
    BarType.GEOMETRIC_MEAN: GeometricMeanBars,
    BarType.MEDIAN: MedianBars,
}
