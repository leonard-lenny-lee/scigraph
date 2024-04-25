from __future__ import annotations

from typing import Callable, Literal, override, TYPE_CHECKING

import numpy as np
from pandas import DataFrame, Index

from scigraph.datatables.abc import DataSet, DataTable
from scigraph.analyses.abc import DescStatsI
from scigraph.config import SG_DEFAULTS

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from scigraph.analyses import DescriptiveStatistics
    from scigraph.analyses._stats import SummaryStatFn
    from scigraph.graphs import ColumnGraph, XYGraph


class ColumnTable(DataTable, DescStatsI):
    
    def __init__(self, values: NDArray) -> None:
        values = self._sanitize_values(values)
        _, n_cols = values.shape

        # Protected attributes
        self._values = values
        self._n_datasets = n_cols
        self._dataset_names = self._default_names(n_cols)

        self.x_title: str = SG_DEFAULTS["datatables.column.x_title"]
        self.y_title: str = SG_DEFAULTS["datatables.column.y_title"]

    @override
    def as_df(self) -> DataFrame:
        return DataFrame(self._values, columns=Index(self._dataset_names))

    @property
    @override
    def dataset_ids(self) -> list[str]:
        return self._dataset_names

    @override
    def get_dataset(self, name: str) -> DataSet:
        try:
            idx = self._dataset_names.index(name)
        except ValueError as e:
            raise ValueError(f"{name} is not a dataset name.") from e
        return DataSet(
            x=None,
            y=self._values[:, idx]
        )

    @property
    def values(self) -> NDArray:
        return self._values

    @property
    def n_datasets(self) -> int:
        return self._n_datasets

    @property
    def dataset_names(self) -> set[str]:
        return set(self._dataset_names)

    @dataset_names.setter
    def dataset_names(self, names: list[str]) -> None:
        contains_duplicates = len(names) != len(set(names))
        incorrect_len = len(names) != self._n_datasets

        if contains_duplicates:
            raise ValueError("Duplicate dataset names not allowed.")
        if incorrect_len:
            raise ValueError("Inappropriate number of names provided")

        self._dataset_names = names

    def _reduce_by_column(self, f: Callable[..., float]) -> NDArray:
        return np.apply_along_axis(f, axis=0, arr=self._values)

    ## Graph factories ##
    
    def create_column_graph(
        self,
        direction: Literal["vertical", "horizontal"]
    ) -> ColumnGraph:
        from scigraph.graphs import ColumnGraph
        return ColumnGraph(self, direction)

    def create_xy_graph(self) -> XYGraph:
        from scigraph.graphs import XYGraph
        return XYGraph.from_column_table(self)

    ## Analysis factories and implementations ##

    def descriptive_statistics(self, *stats: str) -> DescriptiveStatistics:
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
