from __future__ import annotations

__all__ = ["DescriptiveStatistics"]

from typing import Literal, override, TYPE_CHECKING

from scigraph.analyses.abc import Analysis, DescStatsI
from scigraph.analyses._stats import get_summary_statistic_fn
from scigraph._options import SummaryStatistic, DescStatsSubColPolicy as Policy

if TYPE_CHECKING:
    from pandas import DataFrame

    from scigraph.datatables.abc import DataTable


class DescriptiveStatistics(Analysis):

    AVAILABLE_STATISTICS = SummaryStatistic.to_strs()

    def __init__(
        self,
        table: DataTable,
        *stats: str,
        subcolumn_policy: Literal["average", "separate", "merge"],
    ) -> None:
        if not isinstance(table, DescStatsI):
            raise TypeError("DataTable not capable of this analysis")
        self._table = table
        self._statistics: list[SummaryStatistic] = []
        self._subcol_policy = Policy.from_str(subcolumn_policy)
        self.add_statistics(*stats)

    def add_statistics(self, *stats: str) -> None:
        for stat in stats:
            self._statistics.append(
                SummaryStatistic.from_str(stat)
            )

    @override
    def analyze(self) -> DataFrame:
        fns = [get_summary_statistic_fn(fn) for fn in self._statistics]
        match self._subcol_policy:
            case Policy.AVERAGE:
                out = self._table._desc_stats_average(*fns)
            case Policy.SEPARATE:
                out = self._table._desc_stats_average(*fns)
            case Policy.MERGE:
                out = self._table._desc_stats_merge(*fns)
        return out

    @property
    @override
    def table(self) -> DataTable:
        return self._table
