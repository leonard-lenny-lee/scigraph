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
    """A two-factor table with rows, datasets, and replicate subcolumns."""

    def __init__(
        self,
        values: ArrayLike,
        n_replicates: int,
        n_datasets: int,
        dataset_names: Optional[list[str]] = None,
        row_names: Optional[list[str]] = None,
    ) -> None:
        """Create a grouped table from a two-dimensional numeric array.

        Consecutive columns belong to a dataset and represent its replicates.
        """
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
            row_names = self._default_names(n_rows, "Row", numeric_suffix=True)

        self.dataset_names = dataset_names
        self.row_names = row_names

        self.x_title: str = SG_DEFAULTS["datatables.grouped.x_title"]
        self.y_title: str = SG_DEFAULTS["datatables.grouped.y_title"]

    @override
    def as_df(self) -> DataFrame:
        """Return values with row names and dataset/replicate column labels."""
        columns = MultiIndex.from_product(
            (self._dataset_names, range(1, self._n_replicates + 1))
        )
        return DataFrame(self._values, Index(self._row_names), columns)

    @property
    @override
    def dataset_ids(self) -> list[str]:
        """Return dataset labels in their display order."""
        return self._dataset_names

    @override
    def get_dataset(self, name: str) -> DataSet:
        """Return all replicate columns associated with ``name``."""
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
        """Return the underlying numeric values."""
        return self._values

    @property
    def dataset_names(self) -> tuple[str, ...]:
        """Return immutable dataset labels in display order."""
        return tuple(self._dataset_names)

    @dataset_names.setter
    def dataset_names(self, names: list[str]) -> None:
        """Replace dataset labels after validation."""
        self._verify_names(names, self._n_datasets)
        self._dataset_names = names

    @property
    def row_names(self) -> tuple[str, ...]:
        """Return immutable row labels in display order."""
        return tuple(self._row_names)

    @row_names.setter
    def row_names(self, names: list[str]) -> None:
        """Replace row labels after validation."""
        self._verify_names(names, self.nrows)
        self._row_names = names

    @override
    def _set_normalize_values(self, val: NDArray) -> None:
        assert val.shape == self.values.shape
        self._values = val

    @override
    @classmethod
    def from_dataframe(cls, df: DataFrame, **_) -> DataTable:
        """Construct a grouped table from a two-level-column DataFrame.

        The first column level supplies dataset names and the second level
        supplies replicate labels. Every dataset must have the same number of
        replicate columns.
        """
        if df.columns.nlevels != 2:
            raise ValueError(
                "GroupedTable.from_dataframe requires a two-level column index."
            )

        dataset_names = list(df.columns.get_level_values(0).unique())
        replicate_counts = [
            int((df.columns.get_level_values(0) == name).sum())
            for name in dataset_names
        ]
        if not replicate_counts or len(set(replicate_counts)) != 1:
            raise ValueError("Each dataset must have the same number of replicates.")

        row_names = [str(name) for name in df.index]
        return cls(
            df.to_numpy(),
            n_replicates=replicate_counts[0],
            n_datasets=len(dataset_names),
            dataset_names=dataset_names,
            row_names=row_names,
        )

    ## Graph factories ##

    def create_grouped_graph(
        self,
        direction: Literal["vertical", "horizontal"],
        grouping: Literal["interleaved", "separated", "stacked"],
    ) -> GroupedGraph:
        """Create a grouped graph bound to this table."""
        from scigraph.graphs import GroupedGraph

        return GroupedGraph(self, direction, grouping)

    ## Analysis factories and implementations ##

    def row_statistics(
        self, scope: Literal["row", "dataset"], *stats: str
    ) -> RowStatistics:
        """Create a row-statistics analysis for this table."""
        from scigraph.analyses import RowStatistics

        return RowStatistics(self, scope, *stats)

    @override
    def _row_statistics_by_row(self, *fns: SummaryStatFn) -> DataFrame:
        return RowStatsI._row_reduction(self.as_df(), *fns)

    @override
    def _row_statistics_by_dataset(self, *fns: SummaryStatFn) -> DataFrame:
        return RowStatsI._dataset_reduction(self.as_df(), *fns)
