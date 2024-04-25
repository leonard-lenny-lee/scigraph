from __future__ import annotations

from typing import Literal, Optional, override, TYPE_CHECKING

import numpy as np
from pandas import DataFrame, Index, MultiIndex

from scigraph.datatables.abc import DataTable, DataSet
from scigraph.analyses.abc import RowStatsI
from scigraph.config import SG_DEFAULTS

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike

    from scigraph.analyses._stats import SummaryStatFn
    from scigraph.analyses import RowStatistics
    from scigraph.graphs import XYGraph, ColumnGraph


class XYTable(DataTable, RowStatsI):
    """
    Each point is definted by an X and Y coordinate.
    """

    def __init__(
        self,
        values: ArrayLike,
        n_x_replicates: int,
        n_y_replicates: int,
        n_datasets: int,
        dataset_names: Optional[list[str]] = None
    ) -> None:
        values = self._sanitize_values(values)
        expected_ncols = n_x_replicates + n_y_replicates * n_datasets
        if expected_ncols != (n_cols := values.shape[1]):
            raise ValueError(f"Expected {expected_ncols}, found {n_cols}")

        # Protected attributes
        self._values = values
        self._n_x_replicates = n_x_replicates
        self._n_y_replicates = n_y_replicates
        self._n_datasets = n_datasets

        if dataset_names is None:
            dataset_names = self._default_names(n_datasets)
        self._dataset_names = dataset_names

        self.x_title: str = SG_DEFAULTS["datatables.xy.x_title"]
        self.y_title: str = SG_DEFAULTS["datatables.xy.y_title"]

    @override
    def as_df(self) -> DataFrame:
        return DataFrame(self._values, columns=self._columns())

    @property
    @override
    def dataset_ids(self) -> list[str]:
        return self._dataset_names

    @override
    def get_dataset(self, name: str) -> DataSet:
        if name not in self._dataset_index_map:
            raise KeyError(f"{name} is not a dataset name.")

        dataset_idx = self._dataset_index_map[name]
        start_col = dataset_idx * self._n_y_replicates
        end_col = start_col + self._n_y_replicates

        return DataSet(
            x=self.x_values,
            y=self.y_values[:, start_col:end_col]
       )

    @property
    @override
    def values(self) -> NDArray:
        return self._values

    @property
    def y_values(self) -> NDArray:
        return self._values[:, self._n_x_replicates:]

    @property
    def x_values(self) -> NDArray:
        return self._values[:, :self._n_x_replicates]

    @property
    def n_x_replicates(self) -> int:
        return self._n_x_replicates

    @property
    def n_y_replicates(self) -> int:
        return self._n_y_replicates

    @property
    def n_datasets(self) -> int:
        return self._n_datasets

    @property
    def dataset_names(self) -> set[str]:
        return set(self._dataset_names)

    @dataset_names.setter
    def dataset_names(self, names: list[str]) -> None:
        self._verify_names(names, self._n_datasets)
        self._dataset_names = names
        self._generate_dataset_index_map()

    def _columns(self) -> MultiIndex:
        tuples = []
        for n in range(self._n_x_replicates):
            tuples.append((self.x_title, n + 1))
        for dataset in self._dataset_names:
            for n in range(self._n_y_replicates):
                tuples.append((dataset, n + 1))
        return MultiIndex.from_tuples(tuples)

    def _generate_dataset_index_map(self) -> None:
        assert len(self._dataset_names) == self._n_datasets

        self._dataset_index_map: dict[str, int] = dict(
            zip(self._dataset_names, range(self._n_datasets))
        )
    
    ## Graph factories ##

    def create_xy_graph(self) -> XYGraph:
        from scigraph.graphs import XYGraph
        return XYGraph(self)

    def create_column_graph(self, *args, **kwargs) -> ColumnGraph:
        from scigraph.graphs import ColumnGraph
        return ColumnGraph.from_xy_table(self, *args, **kwargs)

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
        out = [np.apply_along_axis(fn, axis=1, arr=self.y_values) for fn in fns]
        out = np.vstack(out).T
        columns = Index([fn.__name__ for fn in fns], name=self.y_title)
        index = Index(np.mean(self.x_values, axis=1), name=self.x_title)
        return DataFrame(out, index, columns)

    @override
    def _row_statistics_by_dataset(self, *fns: SummaryStatFn) -> DataFrame:
        out = self.as_df().T.groupby(level=0).aggregate(fns)
        names = [fn.__name__ for fn in fns]
        out = out.reorder_levels([1, 0], axis=1)[names].T  # type: ignore
        if len(fns) == 1:
            out = out.loc[names[0]]
        out = out.reindex(columns=self._columns().unique(level=0))
        return out
