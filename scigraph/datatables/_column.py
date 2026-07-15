from __future__ import annotations

from typing import Callable, Literal, Optional, override, TYPE_CHECKING

import numpy as np
from pandas import DataFrame, Index

from scigraph.datatables.abc import DataSet, DataTable
from scigraph.analyses.abc import DescStatsI
from scigraph.config import SG_DEFAULTS

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from scigraph.analyses import DescriptiveStatistics
    from scigraph.analyses._stats import SummaryStatFn
    from scigraph.graphs import ColumnGraph, XYGraph


class ColumnTable(DataTable, DescStatsI):
    """A one-factor table in which each column is an independent dataset."""

    def __init__(
        self, values: ArrayLike, dataset_names: Optional[list[str]] = None
    ) -> None:
        """Create a column table from a two-dimensional numeric array.

        Args:
            values: Rows of observations and one column per dataset.
            dataset_names: Optional labels for the dataset columns.
        """
        values = self._sanitize_values(values)
        _, n_cols = values.shape

        # Protected attributes
        self._values = values
        self._n_datasets = n_cols

        if dataset_names is None:
            dataset_names = self._default_names(n_cols)
        self._dataset_names = dataset_names

        self.x_title: str = SG_DEFAULTS["datatables.column.x_title"]
        self.y_title: str = SG_DEFAULTS["datatables.column.y_title"]

    @override
    def as_df(self) -> DataFrame:
        """Return values as a DataFrame labelled with dataset names."""
        return DataFrame(self._values, columns=Index(self._dataset_names))

    @property
    @override
    def dataset_ids(self) -> list[str]:
        """Return dataset labels in their display order."""
        return self._dataset_names

    @override
    def get_dataset(self, name: str) -> DataSet:
        """Return the observations belonging to ``name``."""
        try:
            idx = self._dataset_names.index(name)
        except ValueError as e:
            raise ValueError(f"{name} is not a dataset name.") from e
        return DataSet(x=None, y=self._values[:, idx])

    @property
    def values(self) -> NDArray:
        """Return the underlying numeric values."""
        return self._values

    @property
    def n_datasets(self) -> int:
        """Number of dataset columns."""
        return self._n_datasets

    @property
    def dataset_names(self) -> set[str]:
        """Return the unique dataset labels as a set."""
        return set(self._dataset_names)

    @dataset_names.setter
    def dataset_names(self, names: list[str]) -> None:
        """Replace dataset labels after validating their number and uniqueness."""
        self._verify_names(names, self._n_datasets)
        self._dataset_names = names

    def _reduce_by_column(self, f: Callable[..., float]) -> NDArray:
        return np.apply_along_axis(f, axis=0, arr=self._values)

    @override
    def _set_normalize_values(self, val: NDArray) -> None:
        assert self.values.shape == val.shape
        self._values = val

    @override
    @classmethod
    def from_dataframe(cls, df: DataFrame, **_) -> DataTable:
        """Construct a ColumnTable from a pandas DataFrame.

        First level of the column are taken to be the dataset names.
        """
        dataset_names = df.columns.get_level_values(0)
        return cls(df.values, list(dataset_names))

    ## Graph factories ##

    def create_column_graph(
        self, direction: Literal["vertical", "horizontal"]
    ) -> ColumnGraph:
        """Create a column graph bound to this table."""
        from scigraph.graphs import ColumnGraph

        return ColumnGraph(self, direction)

    def create_xy_graph(self) -> XYGraph:
        """Create an XY graph using row positions as x values."""
        from scigraph.graphs import XYGraph

        return XYGraph.from_column_table(self)

    ## Analysis factories and implementations ##

    def descriptive_statistics(self, *stats: str) -> DescriptiveStatistics:
        """Create a descriptive-statistics analysis for each dataset."""
        from scigraph.analyses import DescriptiveStatistics

        return DescriptiveStatistics(self, *stats, subcolumn_policy="separate")

    @override
    def _desc_stats_separate(self, *fns: SummaryStatFn) -> DataFrame:
        out = [np.apply_along_axis(fn, axis=0, arr=self.values) for fn in fns]
        out = np.vstack(out)
        columns = Index(self.dataset_ids, name=self.x_title)
        index = Index([fn.__name__ for fn in fns])
        return DataFrame(out, index, columns)

    @override
    def _desc_stats_average(self, *fns: SummaryStatFn) -> DataFrame:
        return self._desc_stats_separate(*fns)

    @override
    def _desc_stats_merge(self, *fns: SummaryStatFn) -> DataFrame:
        return self._desc_stats_separate(*fns)
