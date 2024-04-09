from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self, TYPE_CHECKING

from scigraph.styles._plot_properties import (
    generate_plot_prop_cycle,
    PlotProperties
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from scigraph.datatables.abc import DataTable
    from scigraph.analyses.abc import GraphableAnalysis
    from scigraph.graphs import XYGraph


class Graph[T: DataTable, U: GraphableAnalysis](ABC):

    @abstractmethod
    def link_table(self, table: T) -> None:
        self.table = table

    @abstractmethod
    def link_analysis(self, analysis: U) -> Self: ...

    @abstractmethod
    def draw(self, ax: Axes | None) -> Axes: ...

    @property
    def plot_properties(self) -> dict[str, PlotProperties]:
        if not hasattr(self, "_plot_properties"):
            self._compile_plot_properties()
        return self._plot_properties

    def _compile_plot_properties(self) -> None:
        n_datasets = len(self.table.dataset_ids)
        prop_cycle = generate_plot_prop_cycle(n_datasets)
        self._plot_properties = dict(zip(self.table.dataset_ids, prop_cycle))


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
    ) -> None:
        pass
