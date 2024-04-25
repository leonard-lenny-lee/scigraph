from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from matplotlib.axes import Axes

if TYPE_CHECKING:
    from pandas import DataFrame

    from scigraph.analyses._stats import SummaryStatFn
    from scigraph.datatables.abc import DataTable
    from scigraph.graphs.abc import Graph


class Analysis[T: DataTable](ABC):

    @property
    @abstractmethod
    def table(self) -> T: ...

    @abstractmethod
    def analyze(self) -> Any: ...


class GraphableAnalysis[T: DataTable, G: Graph](Analysis[T], ABC):

    @abstractmethod
    def draw(
        self,
        graph: G,
        ax: Axes,
        *args,
        **kwargs,
    ) -> None: ...


class RowStatsI(ABC):

    @abstractmethod
    def _row_statistics_by_row(self, *fns: SummaryStatFn) -> DataFrame: ...

    @abstractmethod
    def _row_statistics_by_dataset(self, *fns: SummaryStatFn) -> DataFrame: ...


class DescStatsI(ABC):

    @abstractmethod
    def _desc_stats_average(self, *fns: SummaryStatFn) -> DataFrame: ...

    @abstractmethod
    def _desc_stats_separate(self, *fns: SummaryStatFn) -> DataFrame: ...

    @abstractmethod
    def _desc_stats_merge(self, *fns: SummaryStatFn) -> DataFrame: ...
