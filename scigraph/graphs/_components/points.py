from __future__ import annotations

from abc import abstractmethod
from typing import override, TYPE_CHECKING
import logging

from matplotlib.axes import Axes
import numpy as np
from pandas import DataFrame

from ..abc import TypeChecked, Artist

if TYPE_CHECKING:
    from scigraph.graphs import XYGraph


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

    @abstractmethod
    def _prepare_xy(
        self,
        graph: XYGraph,
    ) -> DataFrame:
        """Aggregate replicates in dataset"""
        pass

    @classmethod
    def from_str(cls, s: str) -> Points | None:
        if s in _FACTORY_MAP:
            return _FACTORY_MAP[s]()
        return None


class MeanPoints(Points):

    @override
    def _prepare_xy(
        self,
        graph: XYGraph,
    ) -> DataFrame:
        return graph.table.summarize("mean")

    @override
    @classmethod
    def _compatible_types(cls) -> set[str]:
        return {"mean"}


class GeometricMeanPoints(Points):

    @override
    def _prepare_xy(
        self,
        graph: XYGraph,
    ) -> DataFrame:
        return graph.table.summarize("geometric mean")

    @override
    @classmethod
    def _compatible_types(cls) -> set[str]:
        return {"geometric mean"}

class MedianPoints(Points):

    @override
    def _prepare_xy(
        self,
        graph: XYGraph,
    ) -> DataFrame:
        return graph.table.summarize("median")

    @override
    @classmethod
    def _compatible_types(cls) -> set[str]:
        return {"median"}


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
    def _prepare_xy(
        self,
        graph: XYGraph,
    ) -> DataFrame:
        raise NotImplementedError

    @override
    @classmethod
    def _compatible_types(cls) -> set[str]:
        return {"individual"}


_FACTORY_MAP: dict[str, type[Points]] = {
    "mean": MeanPoints,
    "geometric mean": GeometricMeanPoints,
    "median": MedianPoints,
    "individual": IndividualPoints,
}
