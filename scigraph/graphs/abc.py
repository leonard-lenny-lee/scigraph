from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from scigraph.styles._plot_properties import (
    generate_plot_prop_cycle,
    PlotProperties
)


class Graph(ABC):

    @abstractmethod
    def link_table(self, table) -> None: ...

    @abstractmethod
    def link_analysis(self, analysis) -> None: ...

    @abstractmethod
    def draw(self, ax) -> None: ...

    def _init_dataset_plot_props(
        self,
        dataset_labels: list[str]
    ) -> dict[str, PlotProperties]:
        prop_cycle = generate_plot_prop_cycle(len(dataset_labels))
        return dict(zip(dataset_labels, prop_cycle))


class GraphTypeCheckComponent(ABC):

    @classmethod
    def check_compatible(cls, other: Self) -> bool:
        return len(cls.TYPES & other.TYPES) > 0

    @classmethod
    @property
    @abstractmethod
    def TYPES(cls) -> set[str]: ...
