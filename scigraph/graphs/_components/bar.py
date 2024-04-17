from __future__ import annotations

from abc import abstractmethod
from typing import Never, Self, override, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from scigraph.graphs.abc import Artist, TypeChecked
from scigraph._options import (
    GraphType,
    ColumnGraphDirection,
    ColumnGraphSubtype,
    BarType,
)
import scigraph.analyses._agg as agg

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from scigraph.graphs import ColumnGraph


class Bars(Artist, TypeChecked):
    
    @override
    def draw_xy(self, *args, **kwargs) -> Never:
        raise NotImplementedError

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
        if graph._direction is ColumnGraphDirection.VERTICAL:
            ax.bar(x, y, *args, **kwargs)
        else:  # Horizontal
            ax.barh(x, y, *args, **kwargs)

    @abstractmethod
    def _prepare_column(self, graph: ColumnGraph) -> NDArray: ...

    @classmethod
    def from_opt(cls, opt: BarType) -> Self:
        return _FACTORY_MAP[opt]()


class MeanBars(Bars):

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(agg.Basic.mean)

    @override
    @classmethod
    def _compatible_types(cls) -> set[TypeChecked.Type]:
        return {
            TypeChecked.Type(GraphType.COLUMN, ColumnGraphSubtype.MEAN),
            TypeChecked.Type(GraphType.COLUMN, ColumnGraphSubtype.INDIVIDUAL),
            TypeChecked.Type(GraphType.COLUMN, ColumnGraphSubtype.SWARM),
        }


class GeometricMeanBars(Bars):

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(agg.Advanced.geometric_mean)

    @override
    @classmethod
    def _compatible_types(cls) -> set[TypeChecked.Type]:
        return {
            TypeChecked.Type(GraphType.COLUMN, ColumnGraphSubtype.GEOMETRIC_MEAN),
            TypeChecked.Type(GraphType.COLUMN, ColumnGraphSubtype.INDIVIDUAL),
            TypeChecked.Type(GraphType.COLUMN, ColumnGraphSubtype.SWARM),
        }


class MedianBars(Bars):

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(agg.Basic.median)

    @override
    @classmethod
    def _compatible_types(cls) -> set[TypeChecked.Type]:
        return {
            TypeChecked.Type(GraphType.COLUMN, ColumnGraphSubtype.MEDIAN),
            TypeChecked.Type(GraphType.COLUMN, ColumnGraphSubtype.INDIVIDUAL),
            TypeChecked.Type(GraphType.COLUMN, ColumnGraphSubtype.SWARM),
        }


_FACTORY_MAP = {
    BarType.MEAN: MeanBars,
    BarType.GEOMETRIC_MEAN: GeometricMeanBars,
    BarType.MEDIAN:  MedianBars,
}
