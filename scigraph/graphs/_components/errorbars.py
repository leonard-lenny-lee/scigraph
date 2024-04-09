from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, override, Self, TYPE_CHECKING

from numpy.typing import NDArray
from pandas import DataFrame
from matplotlib.axes import Axes

from scigraph.graphs.abc import Artist, TypeChecked

if TYPE_CHECKING:
    from scigraph.graphs import XYGraph


class ErrorBars(Artist, TypeChecked, ABC):

    @override
    def draw_xy(
        self,
        graph: XYGraph,
        ax: Axes,
        *args,
        **kwargs,
    ) -> None:
        draw_x_errors = graph.table.n_x_replicates > 1
        draw_y_errors = graph.table.n_y_replicates > 1

        if not draw_x_errors and not draw_y_errors:
            return

        ori, err = self._prepare_xy(graph)
        x = ori[graph.table.x_title]
        x_err = err[graph.table.x_title] if draw_x_errors else None

        for id in graph.table.dataset_ids:
            y = ori[id]
            y_err = err[id] if draw_y_errors else None
            ax.errorbar(x, y, xerr=x_err, yerr=y_err,
                        *args, **kwargs, marker="", ls="")

    @abstractmethod
    def _prepare_xy(
        self,
        graph: XYGraph,
    ) -> tuple[DataFrame, Errors]: ...

    @classmethod
    def from_str(cls, s: str) -> Self | None:
        if s in _FACTORY_MAP:
            return _FACTORY_MAP[s]()
        return None


class SDErrorBars(ErrorBars):

    @override
    def _prepare_xy(
        self,
        graph: XYGraph
    ) -> tuple[DataFrame, Errors]:
        ori = graph.table.summarize("mean")
        err = graph.table.summarize("sd")
        return ori, err

    @override
    @classmethod
    def _compatible_types(cls) -> set[str]:
        return {"mean"}


class SEMErrorBars(ErrorBars):

    @override
    def _prepare_xy(
        self,
        graph: XYGraph
    ) -> tuple[DataFrame, Errors]:
        ori = graph.table.summarize("mean")
        err = graph.table.summarize("sem")
        return ori, err

    @override
    @classmethod
    def _compatible_types(cls) -> set[str]:
        return {"mean"}


class CI95ErrorBars(ErrorBars):

    @override
    def _prepare_xy(
        self,
        graph: XYGraph
    ) -> tuple[DataFrame, Errors]:
        ori = graph.table.summarize(graph._graph_t)
        err = graph.table.summarize("ci95")
        return ori, err

    @override
    @classmethod
    def _compatible_types(cls) -> set[str]:
        return {"mean", "geometric mean", "median"}


class RangeErrorBars(ErrorBars):

    TYPES = {"mean", "median"}

    @override
    def _prepare_xy(
        self,
        graph: XYGraph
    ) -> tuple[DataFrame, Errors]:
        ori = graph.table.summarize(graph._graph_t)
        lower = ori - graph.table.summarize("min")
        upper = graph.table.summarize("max") - ori
        err = {col: (lower[col].values, upper[col].values) for col in ori}
        return ori, err

    @override
    @classmethod
    def _compatible_types(cls) -> set[str]:
        return {"mean", "median"}


class GeometricSDErrorBars(ErrorBars):

    TYPES = {"geometric mean"}

    @override
    def _prepare_xy(
        self,
        graph: XYGraph
    ) -> tuple[DataFrame, Errors]:
        ori = graph.table.summarize("mean")
        err = graph.table.summarize("geometric sd")
        return ori, err


_FACTORY_MAP = {
    "sd": SDErrorBars,
    "sem": SEMErrorBars,
    "ci95": CI95ErrorBars,
    "range": RangeErrorBars,
    "geometric sd": GeometricSDErrorBars,
}

type Errors = DataFrame | Mapping[str, tuple[NDArray, NDArray]]
