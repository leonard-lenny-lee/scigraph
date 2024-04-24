from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from matplotlib.axes import Axes

from scigraph.analyses._stats import (
    SummaryStatArg, SummaryStatFn, get_summary_statistic_fn
)

if TYPE_CHECKING:
    from pandas import DataFrame

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


class RowStatisticsI(ABC):

    @abstractmethod
    def row_statistics_by_row(
        self,
        fns: SummaryStatArg | list[SummaryStatArg],
    ) -> DataFrame: ...

    @abstractmethod
    def row_statistics_by_dataset(
        self,
        fns: SummaryStatArg | list[SummaryStatArg],
    ) -> DataFrame: ...

    def _row_statistics_normalize_fn_args(
        self,
        fns: SummaryStatArg | list[SummaryStatArg],
    ) -> list[SummaryStatFn]:
        if not isinstance(fns, list):
            fns = [fns]
        out = [get_summary_statistic_fn(fn) if isinstance(fn, str) else fn
               for fn in fns]
        return out
