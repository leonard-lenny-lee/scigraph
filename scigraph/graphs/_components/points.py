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

    @classmethod
    def plot_group(
        cls,
        x: NDArray,
        y: NDArray,
        ax: plt.Axes,
        *args,
        **kwargs
    ) -> Any:
        points = cls._prepare_xy(x, y)
        artist, = ax.plot(points.x, points.y, *args, **kwargs, ls="")
        return artist, points

    @classmethod
    @abstractmethod
    def _prepare_xy(
        cls,
        x: NDArray,
        y: NDArray
    ) -> PointCoordinates: ...


class MeanPoints(Points):

    TYPES = {"mean"}

    @override
    @classmethod
    def _prepare_xy(
        cls,
        x: NDArray,
        y: NDArray
    ) -> PointCoordinates:
        if y.ndim > 1:
            y = y.mean(axis=1)
        return PointCoordinates(x, y)


class GeometricMeanPoints(Points):

    TYPES = {"geometric mean"}

    @override
    @classmethod
    def _prepare_xy(
        cls,
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
    @classmethod
    def _prepare_xy(
        cls,
        x: NDArray,
        y: NDArray
    ) -> PointCoordinates:
        if y.ndim > 1:
            y = y.median(axis=1)
        return PointCoordinates(x, y)


class IndividualPoints(Points):

    TYPES = {"individual"}

    @override
    @classmethod
    def _prepare_xy(
        cls,
        x: NDArray,
        y: NDArray
    ) -> PointCoordinates:
        if y.ndim > 1:
            x = x.flatten().tile(y.shape[1])
            y = y.T.flatten()
        return PointCoordinates(x, y)


_LITERAL_POINTS_MAP: dict[str, type[Points]] = {
    "mean": MeanPoints,
    "geometric mean": GeometricMeanPoints,
    "median": MedianPoints,
    "individual": IndividualPoints,
}


def map_arg_to_points(arg: str) -> type[Points] | None:
    if arg in _LITERAL_POINTS_MAP:
        return _LITERAL_POINTS_MAP[arg]
    return None
