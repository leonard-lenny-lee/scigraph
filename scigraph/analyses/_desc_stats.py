from __future__ import annotations

from typing import TYPE_CHECKING, override

from scigraph.analyses.abc import Analysis
from scigraph.analyses._stats import get_summary_statistic_fn
from scigraph._options import SummaryStatistic

if TYPE_CHECKING:
    from pandas import DataFrame

    from scigraph.analyses.abc import DescriptiveStatisticsI
    from scigraph.datatables.abc import DataTable

    class DataTable_DS(DataTable, DescriptiveStatisticsI): ...


class DescriptiveStatistics(Analysis):

    AVAILABLE_STATISTICS = SummaryStatistic.to_strs()

    def __init__(self, table: DataTable_DS, *stats: str) -> None:
        self._table = table
        self._statistics: list[SummaryStatistic] = []
        self.add_statistics(*stats)

    def add_statistics(self, *stats: str) -> None:
        for stat in stats:
            self._statistics.append(
                SummaryStatistic.from_str(stat)
            )

    @override
    def analyze(self) -> DataFrame:
        fns = [get_summary_statistic_fn(fn) for fn in self._statistics]
        return self.table.descriptive_statistics(*fns)

    @property
    @override
    def table(self) -> DataTable_DS:
        return self._table
