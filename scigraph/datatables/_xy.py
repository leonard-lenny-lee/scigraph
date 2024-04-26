from __future__ import annotations

from typing import Literal, Optional, override, TYPE_CHECKING

import numpy as np
import pandas as pd

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
    A table in which each point is defined by an X and Y coordinate.

    Attributes:
        values: The underlying NumPy array holding the raw data.
        x_values: A view of the NumPy array containing only the X subcolumns.
        y_values: A view of the NumPy array containing only the Y subcolumns.
        nrows: The number of rows in the data array.
        ncols: The number of columns in the data array.
        n_x_replicates: The number of replicates which define X.
        n_y_replicates: The number of replicates which define each Y column.
        n_datasets: The number of Y columns.
        dataset_names: The names of the the Y columns.
        dataset_ids: The names of the the Y columns, alias for dataset_names.
    """

    def __init__(
        self,
        values: ArrayLike,
        n_x_replicates: int,
        n_y_replicates: int,
        n_datasets: int,
        dataset_names: Optional[list[str]] = None,
        x_title: Optional[str] = None,
        y_title: Optional[str] = None,
    ) -> None:
        """Create an XYTable from a 2-dimensional array.

        Data in an XYTable is defined by an X and Y coordinate. The first
        column defines the X values, while the remaining columns defines Y
        values. X and Y columns are further divided into subcolumns which
        define replicate values. Errors are computed from replicate values in 
        subcolumns.

        Args:
            values: Array containing the raw data, must be numerical and 
                2-dimensional. This will be coerced into NumPy np.float64.
            n_x_replicates: The number of replicates which define the X column. 
            n_y_replicates: The number of replicates which define each Y column.
            n_datasets: The number of Y columns.
            dataset_names: The name of the Y columns.
            x_title: The name of the X column.
            y_title: The name of the Y columns, this is used in graphing
                functions as the name of Y axis.

        Raises:
            ValueError: If the specified number of replicates and datasets fail
                to match the dimensions of the provided array or if scigraph
                fails to coerce values into a 2-dimensional np.float64 array.

        Example:
            An XYTable constructed from the following parameters:

            * n_x_replicates = 1
            * n_y_replicates = 3
            * n_datasets = 2
            * dataset_names = ["Control", "Treated"]
            * x_title = "Hours"
            * y_title = "Response, %"

             Hours  Control               Treated            
                  1       1      2      3       1     2     3
            0   0.0    45.0   34.0    NaN    34.0  31.0  29.0
            1   6.0    56.0   58.0   61.0    41.0  43.0  42.0
            2  12.0    76.0   72.0   77.0    52.0  55.0  55.0
            3  24.0    81.0   95.0   83.0    63.0  63.0   NaN
            4  48.0    99.0  100.0  104.0    72.0  67.0  81.0
            5  72.0    97.0  110.0  115.0    78.0  87.0   NaN
        """
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

        # Modifiable attributes
        if x_title is None:
            x_title = SG_DEFAULTS["datatables.xy.x_title"]
        if y_title is None:
            y_title = SG_DEFAULTS["datatables.xy.y_title"]

        self.x_title = x_title
        self.y_title = y_title

    @override
    def as_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._values, columns=self._columns())

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

    def _columns(self) -> pd.MultiIndex:
        """Helper for as_df() to generate MultiIndexed columns."""
        tuples = []
        for n in range(self._n_x_replicates):
            tuples.append((self.x_title, n + 1))
        for dataset in self._dataset_names:
            for n in range(self._n_y_replicates):
                tuples.append((dataset, n + 1))
        return pd.MultiIndex.from_tuples(tuples)

    def _generate_dataset_index_map(self) -> None:
        """Initialize the map to map the dataset name to its index position.
        Allows for easy access of an index position of a dataset.
        """
        assert len(self._dataset_names) == self._n_datasets

        self._dataset_index_map: dict[str, int] = dict(
            zip(self._dataset_names, range(self._n_datasets))
        )
    
    ## Graph factories ##

    def create_xy_graph(self, *args, **kwargs) -> XYGraph:
        """Create an XYGraph instance.

        Initializes a XYGraph instance from a XYTable instance and binds itself
        to the new instance.

        Args:
            *args, **kwargs: Any positional or keyword arguments to be passed
                to the XYGraph initializer.

        Returns:
            An XYGraph instance bound by the current XYTable instance.
        """
        from scigraph.graphs import XYGraph
        return XYGraph(self, *args, **kwargs)

    def create_column_graph(self, *args, **kwargs) -> ColumnGraph:
        """Create an ColumnGraph instance.

        Initializes a ColumnGraph instance from a XYTable instance. It marshals
        itself into a ColumnTable format before binding the ColumnTable
        instance to the ColumnGraph instance. This is generally to be avoided
        as ColumnGraphs are unlikely the appropriately represent XY data and
        suggests the XYTable is inappropriate for the underlying data.

        Args:
            *args, **kwargs: Any positional or keyword arguments to be passed
                to the ColumnGraph initializer.

        Returns:
            An ColumnGraph instance bound by the current XYTable instance.
        """
        from scigraph.graphs import ColumnGraph
        return ColumnGraph.from_xy_table(self, *args, **kwargs)

    ## Analysis factories and implementations ##

    def row_statistics(
        self,
        scope: Literal["row", "dataset"],
        *stats: str
    ) -> RowStatistics:
        """Create a RowStatistics instance.

        Initializes a RowStatistics analysis instance from this current XYTable
        instance.

        Args:
            scope: Defines the analysis grouping policy. If "row", summary
                statistics are computed for the entire row. If "dataset", the
                row is separated into dataset columns before computing summary
                statistics
            stats: The summary statistics to be computed. For example, "mean",
                "sd"...., For a complete list, see the class attribute
                RowStatistics.AVAILABLE_STATISTICS

        Returns:
            A RowStatistics instance
        """
        from scigraph.analyses import RowStatistics
        return RowStatistics(self, scope, *stats)

    @override
    def _row_statistics_by_row(self, *fns: SummaryStatFn) -> pd.DataFrame:
        x = np.mean(self.x_values, axis=1)
        df = self.as_df()[self._dataset_names]
        df.index = x
        return RowStatsI._row_reduction(df, *fns)  # type: ignore

    @override
    def _row_statistics_by_dataset(self, *fns: SummaryStatFn) -> pd.DataFrame:
        return RowStatsI._dataset_reduction(self.as_df(), *fns)
