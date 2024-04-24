from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Never, Self, Any, override, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from scigraph.graphs.abc import GraphComponent
from scigraph._options import ColumnGraphDirection, BarType
import scigraph.analyses._stats as sgstats

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from scigraph.graphs import ColumnGraph


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
        is_vertical = graph._direction is ColumnGraphDirection.VERTICAL

        for i, id in enumerate(graph.table.dataset_ids):
            if not self.line_only:  # Draw full bars
                props = graph.plot_properties[id].bar_kw(is_vertical)
                props.update(**self.kw)
                if is_vertical:
                    ax.bar(x[i], y[i], **props)
                else:  # Horizontal
                    ax.barh(x[i], y[i], **props)
            else:  # Draw top line of bar only
                props = graph.plot_properties[id].barl_kw()
                props.update(**self.kw)
                x_ = x[i] - 0.25, x[i] + 0.25
                if is_vertical:
                    ax.hlines(y[i], *x_, **props)
                else:  # Horizontal
                    ax.vlines(y[i], *x_, **props)

    @abstractmethod
    def _prepare_column(self, graph: ColumnGraph) -> NDArray: ...

    @override
    @classmethod
    def from_opt(cls, opt: BarType, kw, **kwargs) -> Self:
        return _FACTORY_MAP[opt](kw, **kwargs)


class MeanBars(Bars):

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(sgstats.Basic.mean)


class GeometricMeanBars(Bars):

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(sgstats.Advanced.geometric_mean)


class MedianBars(Bars):

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(sgstats.Basic.median)


_FACTORY_MAP = {
    BarType.MEAN: MeanBars,
    BarType.GEOMETRIC_MEAN: GeometricMeanBars,
    BarType.MEDIAN:  MedianBars,
}
