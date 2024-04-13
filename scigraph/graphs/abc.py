from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self, TYPE_CHECKING

from scigraph.analyses.abc import GraphableAnalysis
from scigraph.styles._plot_properties import (
    generate_plot_prop_cycle,
    PlotProperties
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from scigraph.datatables.abc import DataTable
    from scigraph.graphs import XYGraph, ColumnGraph


class Graph[T: DataTable](ABC):

    def __init__(self) -> None:
        self._legend_artists: dict[str, list[Any]] = {}
        self._linked_analyses: list[GraphableAnalysis] = []
        self.include_legend = True

    @abstractmethod
    def link_table(self, table: T) -> None:
        self.table: T
        ...

    def link_analysis(self, analysis: GraphableAnalysis) -> Self:
        if not isinstance(analysis, GraphableAnalysis):
            raise TypeError(f"{type(analysis).__name__} is not graphable.")

        if analysis.table is not self.table:
            raise ValueError("Linked analysis must be from the same DataTable")

        self._linked_analyses.append(analysis)
        return self

    @abstractmethod
    def draw(self, ax: Axes | None) -> Axes: ...

    @property
    @abstractmethod
    def _checkcode(self) -> str:
        """Identifier for Component Compatibility Checking"""

    def _check_component_compatibility(
        self,
        component: TypeChecked
    ) -> None:
        if self._checkcode not in component._compatible_types():
            raise TypeError(
                f"{component.__class__.__name__} is incompatible with "
                f"{self._checkcode} {self.__class__.__name__}s."
            )

    @property
    def plot_properties(self) -> dict[str, PlotProperties]:
        if not hasattr(self, "_plot_properties"):
            self._compile_plot_properties()
        return self._plot_properties

    def _compile_plot_properties(self) -> None:
        n_datasets = len(self.table.dataset_ids)
        prop_cycle = generate_plot_prop_cycle(n_datasets)
        self._plot_properties = dict(zip(self.table.dataset_ids, prop_cycle))

    def _compose_legend(self, ax: Axes):
        labels = []
        handles = []
        for k, v in self._legend_artists.items():
            labels.append(k)
            handles.append(tuple(v))

        return ax.legend(handles, labels)

    def _add_legend_artist(self, group: str, artist: Any) -> None:
        if group not in self._legend_artists:
            self._legend_artists[group] = []

        self._legend_artists[group].append(artist)


class TypeChecked(ABC):

    @classmethod
    def check_compatible(cls, other: Self) -> bool:
        return len(cls._compatible_types() & other._compatible_types()) > 0

    @classmethod
    @abstractmethod
    def _compatible_types(cls) -> set[str]: ...


class Artist(ABC):

    @abstractmethod
    def draw_xy(
        self,
        graph: XYGraph,
        ax: Axes,
        *args,
        **kwargs,
    ) -> None: ...

    @abstractmethod
    def draw_column(
        self,
        graph: ColumnGraph,
        ax: Axes,
        *args,
        **kwargs,
    ) -> None: ...
