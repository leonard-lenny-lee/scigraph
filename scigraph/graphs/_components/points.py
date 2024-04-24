from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Never, override, TYPE_CHECKING

from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
import seaborn as sns

from scigraph.graphs.abc import GraphComponent
from scigraph._options import ColumnGraphDirection, PointsType
import scigraph.analyses._stats as sgstats
from scigraph._log import LOG

if TYPE_CHECKING:
    from scigraph.graphs import XYGraph, ColumnGraph


class Points(GraphComponent, ABC):

    @override
    def draw_xy(self, graph: XYGraph, ax: Axes) -> None:
        agg = self._prepare_xy(graph)
        x = agg[graph.table.x_title]

        for id in graph.table.dataset_ids:
            props = graph.plot_properties[id].point_kws()
            y = agg[id]
            props.update(**self.kw)
            artist, = ax.plot(x, y, **props)
            graph._add_legend_artist(id, artist)

    @override
    def draw_column(self, graph: ColumnGraph, ax: Axes) -> None:
        x = np.linspace(0, graph.table.ncols - 1, graph.table.ncols)
        y = self._prepare_column(graph)
        if graph._direction is ColumnGraphDirection.HORIZONTAL:
            x, y = y, x
        
        for i, id in enumerate(graph.table.dataset_ids):
            props = graph.plot_properties[id].point_kws()
            props.update(**self.kw)
            ax.plot(x[i], y[i], **props)

    @abstractmethod
    def _prepare_xy(self, graph: XYGraph, /,) -> DataFrame: ...

    @abstractmethod
    def _prepare_column(self, graph: ColumnGraph, /,) -> NDArray: ...

    @override
    @classmethod
    def from_opt(cls, opt: PointsType, kw, **_) -> Points:
        return _FACTORY_MAP[opt](kw)


class MeanPoints(Points):

    @override
    def _prepare_xy(self, graph: XYGraph) -> DataFrame:
        return graph.table.row_statistics_by_dataset(sgstats.Basic.mean)

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(sgstats.Basic.mean)


class GeometricMeanPoints(Points):

    @override
    def _prepare_xy(self, graph: XYGraph) -> DataFrame:
        return graph.table.row_statistics_by_dataset(
            sgstats.Advanced.geometric_mean
        )

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(sgstats.Advanced.geometric_mean)


class MedianPoints(Points):

    @override
    def _prepare_xy(self, graph: XYGraph) -> DataFrame:
        return graph.table.row_statistics_by_dataset(sgstats.Basic.median)

    @override
    def _prepare_column(self, graph: ColumnGraph) -> NDArray:
        return graph.table._reduce_by_column(sgstats.Basic.median)


class IndividualPoints(Points):

    @override
    def draw_xy(self, graph: XYGraph, ax: Axes) -> None:
        # No aggregation
        x = graph.table.x_values
        if graph.table.n_x_replicates > 1:
            x = x.mean(axis=1)
            LOG.warn(
                "Multiple X values are incompatible with individual points "
                "and have been averaged."
            )
        # Match y series dimensions
        x = np.tile(x.flatten(), graph.table.n_y_replicates)

        for id, dataset in graph.table.datasets_itertuples():
            props = graph.plot_properties[id].point_kws()
            props.update(**self.kw)
            y = dataset.y.T.flatten()
            assert len(x) == len(y)
            artist, = ax.plot(x, y, **props)
            graph._add_legend_artist(id, artist)

    @override
    def draw_column(self, graph: ColumnGraph, ax: Axes) -> None:
        x = np.linspace(0, graph.table.ncols - 1, graph.table.ncols)
        x = np.repeat(x, graph.table.nrows)
        y = graph.table.values.flatten('F')

        if graph._direction is ColumnGraphDirection.HORIZONTAL:
            x, y = y, x
        
        for i, id in enumerate(graph.table.dataset_ids):
            props = graph.plot_properties[id].point_kws()
            props.update(**self.kw)
            x_slice = x[i*graph.table.nrows:(i+1)*graph.table.nrows]
            y_slice = y[i*graph.table.nrows:(i+1)*graph.table.nrows]
            ax.plot(x_slice, y_slice, **props)

    @override
    def _prepare_xy(self, _) -> Never:
        raise NotImplementedError

    @override
    def _prepare_column(self, _) -> Never:
        raise NotImplementedError


class SwarmPoints(Points):

    @override
    def draw_column(self, graph: ColumnGraph, ax: Axes) -> None:
        x = np.linspace(0, graph.table.ncols - 1, graph.table.ncols)
        x = np.repeat(x, graph.table.nrows)
        y = graph.table.values.flatten('F')
        # Use seaborn for swarm
        if graph._direction is ColumnGraphDirection.HORIZONTAL:
            orient = "h"
            x, y = y, x
        else:
            orient = "v"

        for i, id in enumerate(graph.table.dataset_ids):
            props = graph.plot_properties[id]
            s = slice(i * graph.table.nrows, (i + 1) * graph.table.nrows)
            sns.swarmplot(x=x[s], y=y[s], orient=orient, legend=False,
                          ax=ax, **self.kw, marker=props.marker,
                          color=props.color, size=props.markersize)

    @override
    def draw_xy(self, *args, **kwargs) -> Never:
        raise NotImplementedError

    @override
    def _prepare_xy(self, _) -> Never:
        raise NotImplementedError

    @override
    def _prepare_column(self, _) -> Never:
        raise NotImplementedError


_FACTORY_MAP: dict[PointsType, type[Points]] = {
    PointsType.MEAN: MeanPoints,
    PointsType.GEOMETRIC_MEAN: GeometricMeanPoints,
    PointsType.MEDIAN: MedianPoints,
    PointsType.INDIVIDUAL: IndividualPoints,
    PointsType.SWARM: SwarmPoints,
}
