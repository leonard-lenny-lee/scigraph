"""
Artists that connect average of replicates together, or individual replicates
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self, override, TYPE_CHECKING

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from scigraph.graphs.abc import Artist, TypeChecked

if TYPE_CHECKING:
    from scigraph.graphs import XYGraph


class ConnectingLine(Artist, TypeChecked, ABC):

    def __init__(self, join_nan: bool) -> None:
        self.join_nan = join_nan

    @override
    def draw_xy(
        self,
        graph: XYGraph,
        ax: Axes,
        *args,
        **kwargs,
    ) -> None:
        agg = self._prepare_xy(graph)

        for id in graph.table.dataset_ids:
            props = graph.plot_properties[id]
            x = np.array(agg[graph.table.x_title].values).flatten()
            y = np.array(agg[id].values).flatten()
            if self.join_nan:
                # Mask NaN values so there is a continuous joined line
                x, y = self._mask_nan(x, y)
            artist, = ax.plot(x, y, *args, **kwargs,
                              marker="", color=props.color, ls=props.linestyle)
            graph._add_legend_artist(id, artist)

    @abstractmethod
    def _prepare_xy(
        self,
        graph: XYGraph,
    ) -> DataFrame: ...

    def _mask_nan(self, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        assert x.shape == y.shape
        assert x.ndim == 1
        assert y.ndim == 1

        stacked_array = np.vstack((x, y))
        mask = ~np.any(np.isnan(stacked_array), axis=0)
        masked_array = stacked_array[:, mask]
        return masked_array[0], masked_array[1]

    @classmethod
    def from_str(cls, s: str, *args, **kwargs) -> Self | None:
        if s in _FACTORY_MAP:
            return _FACTORY_MAP[s](*args, **kwargs)
        return None


class MeanConnectingLine(ConnectingLine):

    @override
    def _prepare_xy(
        self,
        graph: XYGraph,
    ) -> DataFrame:
        return graph.table.summarize("mean")

    @override
    @classmethod
    def _compatible_types(cls) -> set[str]:
        return {"mean", "individual"}


class GeometricMeanConnectingLine(ConnectingLine):

    @override
    def _prepare_xy(
        self,
        graph: XYGraph,
    ) -> DataFrame:
        return graph.table.summarize("geometric mean")

    @override
    @classmethod
    def _compatible_types(cls) -> set[str]:
        return {"geometric mean", "individual"}


class MedianConnectingLine(ConnectingLine):

    @override
    def _prepare_xy(
        self,
        graph: XYGraph,
    ) -> DataFrame:
        return graph.table.summarize("median")

    @override
    @classmethod
    def _compatible_types(cls) -> set[str]:
        return {"median", "individual"}


class IndividualConnectingLine(ConnectingLine):

    @override
    def draw_xy(
        self,
        graph: XYGraph,
        ax: Axes,
        *args,
        **kwargs,
    ) -> None:
        x = graph.table.x_values.mean(axis=1).flatten()

        for id, data in graph.table.datasets_itertuples():
            props = graph.plot_properties[id]
            for y in data.y.T:
                assert x.shape == y.shape
                if self.join_nan:
                    x_, y = self._mask_nan(x, y)
                else:
                    x_ = x
                ax.plot(x_, y, *args, **kwargs,
                                  marker="", color=props.color, ls=props.linestyle)
            artist = Line2D([], [],
                            marker="", color=props.color, ls=props.linestyle)
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


_FACTORY_MAP = {
    "mean": MeanConnectingLine,
    "geometric mean": GeometricMeanConnectingLine,
    "median": MeanConnectingLine,
    "individual": IndividualConnectingLine,
}
