from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Never, Self, override, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from scigraph.graphs.abc import GraphComponent
from scigraph._options import ColumnGraphDirection, BarType
import scigraph.analyses._agg as agg

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from scigraph.graphs import ColumnGraph


class Bars(GraphComponent, ABC):
    
    def __init__(self, line_only: bool = False) -> None:
        self.line_only = line_only

    @override
    def draw_xy(self, *args, **kwargs) -> Never:
        raise NotImplementedError

    @override
    def draw_column(
        self,
        graph: ColumnGraph,
        ax: Axes,
        *args,
        **kwargs,
    ) -> None:
        x = np.linspace(0, graph.table.ncols - 1, graph.table.ncols)
        y = self._prepare_column(graph)
        if not self.line_only:  # Draw full bars
            if graph._direction is ColumnGraphDirection.VERTICAL:
                ax.bar(x, y, *args, **kwargs)
            else:  # Horizontal
                ax.barh(x, y, *args, **kwargs)
        else:  # Draw top line of bar only
            if graph._direction is ColumnGraphDirection.VERTICAL:
                ax.hlines(y, x - 0.25, x + 0.25, *args, **kwargs)
            else:  # Horizontal
                ax.vlines(y, x - 0.25, x + 0.25, *args, **kwargs)

    @abstractmethod
    def _prepare_column(self, graph: ColumnGraph) -> NDArray: ...

    @override
    @classmethod
    def from_opt(cls, opt: BarType, **kwargs) -> Self:
        return _FACTORY_MAP[opt](**kwargs)


class MeanBars(Bars):

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(agg.Basic.mean)


class GeometricMeanBars(Bars):

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(agg.Advanced.geometric_mean)


class MedianBars(Bars):

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(agg.Basic.median)


_FACTORY_MAP = {
    BarType.MEAN: MeanBars,
    BarType.GEOMETRIC_MEAN: GeometricMeanBars,
    BarType.MEDIAN:  MedianBars,
}
