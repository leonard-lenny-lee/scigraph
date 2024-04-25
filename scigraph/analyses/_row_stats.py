from __future__ import annotations

from typing import Literal, TYPE_CHECKING, override

from scigraph.analyses.abc import Analysis, RowStatsI
from scigraph.analyses._stats import get_summary_statistic_fn
from scigraph._options import RowStatisticsScope, SummaryStatistic

if TYPE_CHECKING:
    from pandas import DataFrame

    from scigraph.datatables.abc import DataTable


class RowStatistics(Analysis):

    AVAILABLE_STATISTICS = SummaryStatistic.to_strs()
    
    def __init__(
        self,
        table: DataTable,
        scope: Literal["row", "dataset"],
        *stats: str,
    ) -> None:
        if not isinstance(table, RowStatsI):
            raise TypeError("Table type does not implement Row Statistics.")
        self._table = table
        self._scope = RowStatisticsScope.from_str(scope)
        self._statistics: list[SummaryStatistic] = []
        self.add_statistics(*stats)

    def add_statistics(self, *stats: str) -> None:
        for stat in stats:
            self._statistics.append(
                SummaryStatistic.from_str(stat)
            )

    @override
    def analyze(self) -> DataFrame:
        fns = [get_summary_statistic_fn(s) for s in self._statistics]
        if self._scope is RowStatisticsScope.DATASET:
            out = self._table._row_statistics_by_dataset(*fns)
        else:  # Entire row
            out = self._table._row_statistics_by_row(*fns)
        return out

    @property
    @override
    def table(self) -> DataTable:
        return self._table
