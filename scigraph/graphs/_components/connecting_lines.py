from abc import ABC, abstractmethod
from typing import Any, NamedTuple, override

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from scigraph.graphs.abc import GraphTypeCheckComponent


class PointCoordinates(NamedTuple):
    x: NDArray
    y: NDArray


class ConnectingLine(GraphTypeCheckComponent, ABC):

    def __init__(self, join_nan: bool) -> None:
        self.join_nan = join_nan

    def plot_group(
        self,
        x: NDArray,
        y: NDArray,
        ax: plt.Axes,
        *args,
        **kwargs
    ) -> Any:
        points = self._prepare_xy(x, y)
        if self.join_nan:
            points = self._mask_nan(points)
        artist, = ax.plot(points.x, points.y, *args, **kwargs, marker="")
        return artist

    @abstractmethod
    def _prepare_xy(
        self,
        x: NDArray,
        y: NDArray
    ) -> PointCoordinates: ...

    def _mask_nan(self, points: PointCoordinates) -> PointCoordinates:
        assert points.x.shape == points.y.shape
        assert points.x.ndim == 1
        assert points.y.ndim == 1

        stacked_array = np.vstack((points.x, points.y))
        mask = ~np.any(np.isnan(stacked_array), axis=0)
        masked_array = stacked_array[:, mask]
        return PointCoordinates(masked_array[0], masked_array[1])


class MeanConnectingLine(ConnectingLine):

    TYPES = {"mean", "individual"}

    @override
    def _prepare_xy(
        self,
        x: NDArray,
        y: NDArray
    ) -> PointCoordinates:
        if y.ndim > 1:
            y = y.mean(axis=1)
        return PointCoordinates(x, y)


class GeometricMeanConnectingLine(ConnectingLine):

    TYPES = {"geometric mean", "individual"}

    @override
    def _prepare_xy(
        self,
        x: NDArray,
        y: NDArray
    ) -> PointCoordinates:
        if y.ndim > 1:
            sample_count = np.count_nonzero(~np.isnan(y), axis=1)
            y = y.prod(axis=1) ** (1 / sample_count)
        return PointCoordinates(x, y)


class MedianConnectingLine(ConnectingLine):

    TYPES = {"median", "individual"}

    @override
    def _prepare_xy(
        self,
        x: NDArray,
        y: NDArray
    ) -> PointCoordinates:
        if y.ndim > 1:
            y = y.median(axis=1)
        return PointCoordinates(x, y)


class IndividualConnectingLine(ConnectingLine):

    TYPES = {"individual"}

    @override
    def plot_group(
        self,
        x: NDArray,
        y: NDArray,
        ax: plt.Axes,
        *args,
        **kwargs,
    ) -> Any:
        for y_ in y.T:
            points = PointCoordinates(x, y_)
            if self.join_nan:
                points = self._mask_nan(points)
            artist, = ax.plot(points.x, points.y, *args, **kwargs, marker="")
        return artist

    @override
    def _prepare_xy(
        self,
        x: NDArray,
        y: NDArray,
    ) -> PointCoordinates:
        raise NotImplementedError


_FACTORY_MAP = {
    "mean": MeanConnectingLine,
    "geometric mean": GeometricMeanConnectingLine,
    "median": MeanConnectingLine,
    "individual": IndividualConnectingLine,
}


def connecting_line_factory_fn(
    ty: str,
    *args,
    **kwargs,
) -> ConnectingLine | None:
    if ty in _FACTORY_MAP:
        return _FACTORY_MAP[ty](*args, **kwargs)
    return None
