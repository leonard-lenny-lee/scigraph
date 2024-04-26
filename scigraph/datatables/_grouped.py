from __future__ import annotations

from typing import Literal, Optional, override, TYPE_CHECKING

from pandas import DataFrame, Index, MultiIndex

from scigraph.datatables.abc import DataSet, DataTable
from scigraph.analyses.abc import RowStatsI
from scigraph.config import SG_DEFAULTS

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike

    from scigraph.analyses._stats import SummaryStatFn
    from scigraph.analyses import RowStatistics
    from scigraph.graphs import GroupedGraph


class GroupedTable(DataTable, RowStatsI):
    """
    Grouped tables have two grouping variables, one defined by columns and the
    other defined by rows.
    """

    def __init__(
        self,
        values: ArrayLike,
        n_replicates: int,
        n_datasets: int,
        dataset_names: Optional[list[str]] = None,
        row_names: Optional[list[str]] = None,
    ) -> None:
        values = self._sanitize_values(values)
        n_rows, n_cols = values.shape
        expected_cols = n_replicates * n_datasets
        if expected_cols != n_cols:
            raise ValueError(f"Expected {expected_cols}, found {n_cols}")

        self._values = values
        self._n_replicates = n_replicates
        self._n_datasets = n_datasets

        if dataset_names is None:
            dataset_names = self._default_names(n_datasets)
        if row_names is None:
            row_names = self._default_names(n_rows, "Row", numeric_suffix= True)

        self.dataset_names = dataset_names
        self.row_names = row_names

        self.x_title: str = SG_DEFAULTS["datatables.grouped.x_title"]
        self.y_title: str = SG_DEFAULTS["datatables.grouped.y_title"]

    @override
    def as_df(self) -> DataFrame:
        columns = MultiIndex.from_product(
            (self._dataset_names, range(1, self._n_replicates + 1))
        )
        return DataFrame(self._values, Index(self._row_names), columns)

    @property
    @override
    def dataset_ids(self) -> list[str]:
        return self._dataset_names

    @override
    def get_dataset(self, name: str) -> DataSet:
        try:
            i = self.dataset_ids.index(name)
        except ValueError as e:
            raise KeyError(f"{name} is not a dataset name") from e

        start_col = i * self._n_replicates
        end_col = start_col + self._n_replicates

        return DataSet(x=None, y=self._values[:, start_col:end_col])

    @property
    @override
    def values(self) -> NDArray:
        return self._values

    @property
    def dataset_names(self) -> tuple[str, ...]:
        return tuple(self._dataset_names)

    @dataset_names.setter
    def dataset_names(self, names: list[str]) -> None:
        self._verify_names(names, self._n_datasets)
        self._dataset_names = names

    @property
    def row_names(self) -> tuple[str, ...]:
        return tuple(self._row_names)

    @row_names.setter
    def row_names(self, names: list[str]) -> None:
        self._verify_names(names, self.nrows)
        self._row_names = names

    ## Graph factories ##

    def create_grouped_graph(
        self,
        direction: Literal["vertical", "horizontal"],
        grouping: Literal["interleaved", "separated", "stacked"],
    ) -> GroupedGraph:
        from scigraph.graphs import GroupedGraph
        return GroupedGraph(self, direction, grouping)

    ## Analysis factories and implementations ##

    def row_statistics(
        self,
        scope: Literal["row", "dataset"],
        *stats: str
    ) -> RowStatistics:
        from scigraph.analyses import RowStatistics
        return RowStatistics(self, scope, *stats)

    @override
    def _row_statistics_by_row(self, *fns: SummaryStatFn) -> DataFrame:
        return RowStatsI._row_reduction(self.as_df(), *fns)

    @override
    def _row_statistics_by_dataset(self, *fns: SummaryStatFn) -> DataFrame:
        return RowStatsI._dataset_reduction(self.as_df(), *fns)
