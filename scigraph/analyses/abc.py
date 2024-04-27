from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from matplotlib.axes import Axes
import numpy as np
from pandas import Index, DataFrame

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

    @property
    def result(self) -> Any:
        if not hasattr(self, "_result"):
            self._result = self.analyze()
        return self._result


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

    @staticmethod
    def _row_reduction(df: DataFrame, *fns: SummaryStatFn) -> DataFrame:
        out = [np.apply_along_axis(fn, axis=1, arr=df.values) for fn in fns]
        out = np.vstack(out).T
        columns = Index([fn.__name__ for fn in fns])
        return DataFrame(out, df.index, columns)

    @staticmethod
    def _dataset_reduction(df: DataFrame, *fns: SummaryStatFn) -> DataFrame:
        assert df.columns.nlevels == 2
        out = df.T.groupby(level=0).aggregate(fns)
        names = [fn.__name__ for fn in fns]
        out = out.reorder_levels([1, 0], axis=1)[names].T  # type: ignore
        if len(fns) == 1:
            out = out.loc[names[0]]
        out = out.reindex(columns=df.columns.unique(level=0))
        return out


class DescStatsI(ABC):

    @abstractmethod
    def _desc_stats_average(self, *fns: SummaryStatFn) -> DataFrame: ...

    @abstractmethod
    def _desc_stats_separate(self, *fns: SummaryStatFn) -> DataFrame: ...

    @abstractmethod
    def _desc_stats_merge(self, *fns: SummaryStatFn) -> DataFrame: ...
