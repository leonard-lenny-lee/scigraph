from abc import ABC, abstractmethod
from typing import override, NamedTuple, Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ..abc import GraphTypeCheckComponent


class PointCoordinates(NamedTuple):
    x: NDArray
    y: NDArray


class Points(GraphTypeCheckComponent, ABC):

    def plot_group(
        self,
        x: NDArray,
        y: NDArray,
        ax: plt.Axes,
        *args,
        **kwargs
    ) -> Any:
        points = self._prepare_xy(x, y)
        artist, = ax.plot(points.x, points.y, *args, **kwargs, ls="")
        return artist, points

    @abstractmethod
    def _prepare_xy(
        self,
        x: NDArray,
        y: NDArray
    ) -> PointCoordinates: ...


class MeanPoints(Points):

    TYPES = {"mean"}

    @override
    def _prepare_xy(
        self,
        x: NDArray,
        y: NDArray
    ) -> PointCoordinates:
        if y.ndim > 1:
            y = y.mean(axis=1)
        return PointCoordinates(x, y)


class GeometricMeanPoints(Points):

    TYPES = {"geometric mean"}

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


class MedianPoints(Points):

    TYPES = {"median"}

    @override
    def _prepare_xy(
        self,
        x: NDArray,
        y: NDArray
    ) -> PointCoordinates:
        if y.ndim > 1:
            y = y.median(axis=1)
        return PointCoordinates(x, y)


class IndividualPoints(Points):

    TYPES = {"individual"}

    @override
    def _prepare_xy(
        self,
        x: NDArray,
        y: NDArray
    ) -> PointCoordinates:
        if y.ndim > 1:
            x = np.tile(x.values.flatten(), y.shape[1])
            y = y.T.flatten()
        return PointCoordinates(x, y)


_FACTORY_MAP: dict[str, type[Points]] = {
    "mean": MeanPoints,
    "geometric mean": GeometricMeanPoints,
    "median": MedianPoints,
    "individual": IndividualPoints,
}


def points_factory_fn(ty: str) -> Points | None:
    if ty in _FACTORY_MAP:
        return _FACTORY_MAP[ty]()
    return None
