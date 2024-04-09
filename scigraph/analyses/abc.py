from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from matplotlib.axes import Axes

if TYPE_CHECKING:
    from scigraph.datatables.abc import DataTable
    from scigraph.graphs.abc import Graph


class Analysis[T: DataTable](ABC):

    @property
    @abstractmethod
    def table(self) -> T:
        pass

    @abstractmethod
    def analyze(self) -> Any:
        pass


class GraphableAnalysis[T: Graph](Analysis, ABC):

    @abstractmethod
    def draw(
        self,
        graph: T,
        ax: Axes,
        *args,
        **kwargs,
    ) -> None:
        ...

