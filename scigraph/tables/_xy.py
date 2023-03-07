"""Contains the XYTable class
"""
from __future__ import annotations
from typing import Optional, Iterable
import warnings
from ._datatable import DataTable, DataFrame, MultiIndex

__all__ = ["XYTable"]


class XYTable(DataTable):

    def __init__(
        self,
        data,
        n_x_replicates: int,
        n_y_groups: int,
        n_y_replicates: int,
        x_name: Optional[str] = None,
        x_units: Optional[str] = None,
        y_name: Optional[str] = None,
        y_units: Optional[str] = None,
        row_names: Optional[Iterable] = None,
        group_names: Optional[Iterable] = None
    ) -> None:
        """An XY table is a graph where every point is defined by both an X
        and a Y value. This kind of data are often fit with linear or nonlinear
        regression.

        Parameters
        ----------
        data : array-like
            An array of the data, including X and y. If provided in a
            DataFrame, the indices are ignored and only the array is used. To
            transform a DataFrame directly into the XYTable, use the method
            XYTable.from_frame
        n_x_replicates : int
            The number of columns which represent the X data. X data is
            always assumed to be aligned left
        n_y_groups : int
            The number of Y groups in the dataset
        n_y_replicates : int
            The number of replicates (columns) per Y group
        x_name : Optional[str], optional
            The name of the X columns, by default None
        row_names : Optional[Iterable], optional
            The names of the rows. If none is provided, rows will be named
            0, 1, 2, 3..., by default None
        group_names : Optional[Iterable], optional
            The names of the Y groups. If none is provided, groups will be
            named A, B, C ... ZX, ZY, ZZ, by default None
        """
        self.data = data
        self._n_x_replicates = n_x_replicates
        self._n_y_groups = n_y_groups
        self._n_y_replicates = n_y_replicates
        self.x_name = x_name
        self.x_units = x_units
        self.y_name = y_name
        self.y_units = y_units
        self.row_names = row_names
        self.group_names = group_names

        super().__init__()

    def reshape(
        self,
        n_x_replicates: int,
        n_y_groups: int,
        n_y_replicates: int
    ) -> XYTable:
        """Reshape the table by changing the specifiers for the number of
        groups, and the number of X and Y replicates. The number of columns
        must match the number of columns in the underlying DataFrame.

        Parameters
        ----------
        n_x_replicates : int
            The number of columns which represent the X data. X data is
            always assumed to be aligned left
        n_y_groups : int
            The number of Y groups in the dataset
        n_y_replicates : int
            The number of replicates (columns) per Y group

        Returns
        -------
        XYTable
            A reference to the object
        """
        # Check that the new specified shape matches shape of data
        expected_columns = n_y_groups * n_y_replicates + n_x_replicates
        super()._check_shape(expected_columns=expected_columns)
        self._n_y_groups = n_y_groups
        self._n_y_replicates = n_y_replicates
        self._n_x_replicates = n_x_replicates
        # Rename groups, if necessary
        n_names = len(self.group_names)
        if n_names > self.n_y_groups:
            # Truncate excess group names
            excess_groups = ", ".join(self.group_names[self.n_y_groups:])
            warnings.warn(
                f"The number of specified Y groups has been reduced."
                f"Excess group names: '{excess_groups}' have been deleted"
            )
            del self.group_names[self.n_y_groups:]
        elif n_names < self.n_y_groups:
            # Add additional group names
            gen_n_names = self.n_y_groups - n_names
            self.group_names += self._auto_name(
                prefix="Group_", n_names=gen_n_names, alpha=True,
                start=65+n_names
            )
        assert len(self.group_names) == self.n_y_groups
        self._set_data()
        return self

    @property
    def n_x_replicates(self) -> int:
        return self._n_x_replicates

    @property
    def n_y_groups(self) -> int:
        return self._n_y_groups

    @property
    def n_y_replicates(self) -> int:
        return self._n_y_replicates

    @property
    def row_names(self) -> Iterable:
        return self._row_names

    @row_names.setter
    def row_names(self, names: Iterable) -> None:
        n_rows, _ = self.values.shape
        self._check_names(names=names, n=n_rows)
        self._row_names = names

    @property
    def group_names(self) -> Iterable:
        return self._group_names

    @group_names.setter
    def group_names(self, names: Iterable) -> None:
        self._check_names(names=names, n=self.n_y_groups)
        self._group_names = names

    @property
    def x_label(self) -> str:
        if self.x_units is None:
            return self.x_name
        return f"{self.x_name} / {self.x_units}"

    @property
    def y_label(self) -> str:
        if self.y_units is None:
            return self.y_name
        return f"{self.y_name} / {self.y_units}"

    @property
    def _grouped(self):
        return self.data.groupby(axis=1, level=0, sort=False)

    @property
    def std(self) -> DataFrame:
        return self._grouped.std(numeric_only=True)

    @property
    def mean(self) -> DataFrame:
        return self._grouped.mean()

    @property
    def median(self) -> DataFrame:
        return self._grouped.median()

    @property
    def min(self) -> DataFrame:
        return self._grouped.min()

    @property
    def max(self) -> DataFrame:
        return self._grouped.max()

    @property
    def count(self) -> DataFrame:
        return self._grouped.count()

    @property
    def sem(self) -> DataFrame:
        return self.std / (self.count ** 0.5)

    def _check_shape(self) -> None:
        """Check that there is a column for each replicate of each y group and
        the replicates for the X group
        """
        expected_cols = self.n_y_groups * self.n_y_replicates + self.n_x_replicates
        super()._check_shape(expected_columns=expected_cols)

    def _init_names(self) -> None:
        """Create the necessary column_names for indexing the DataTable from
        the specified parameters. If no column_names are provided generate a
        default of Group A, B, C, ... ZY, ZZ
        """
        # Generate x_names
        self.x_name = "X" if self.x_name is None else str(self.x_name)
        self.y_name = "Y" if self.y_name is None else str(self.y_name)
        # Generate default column_names if none is provided
        if self.group_names is None:
            self. group_names = self._auto_name(
                prefix="Group_", n_names=self.n_y_groups, alpha=True
            )
            return
        # Validate user input, if provided
        n_rows, _ = self.data.shape
        self._check_names(self.group_names, self.n_y_groups)
        self._check_names(self.row_names, n_rows)

    def _set_data(self) -> None:
        """Construct the MultiIndices for the DataFrame and set attribute
        """
        X_col = [(self.x_name, n) for n in range(1, self.n_x_replicates + 1)]
        y_col = [(group, n) for group in self.group_names
                 for n in range(1, self.n_y_replicates + 1)]
        columns = MultiIndex.from_tuples(X_col + y_col, names=["Group", "n"])

        self.data = DataFrame(
            self.data.values, index=self.row_names, columns=columns
        )

    @classmethod
    def from_frame(cls, df: DataFrame) -> XYTable:
        """Construct an XYTable from a pandas DataFrame while preserving
        the columns and indices. 

        Parameters
        ----------
        df : DataFrame
            DataFrame to be converted into a XYTable

        Returns
        -------
        XYTable
            The XYTable constructed from a DataFrame    
        """
        cls._validate_nested_frame(df)
        # Construct arguments for instance initialisation
        group_sizes = df.groupby(level=0, axis=1, sort=False).size()
        data = df.values
        n_x_replicates = group_sizes[0]
        n_y_groups = len(group_sizes[1:])
        group_size = group_sizes[1]
        row_names = df.index
        x_name, column_names = group_sizes.index[0], list(
            group_sizes.index[1:])
        return cls(data, n_x_replicates, n_y_groups, group_size, x_name,
                   row_names, column_names)
