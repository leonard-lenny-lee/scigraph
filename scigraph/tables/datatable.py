"""Contains the DataTable abstract base class
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, List
from pandas import DataFrame, MultiIndex, to_numeric
from numpy import ndarray


class DataTable(ABC):

    @abstractmethod
    def __init__(self):
        """Default pipeline for validating and instantiating datatable
        """
        self._check_shape()
        self._init_names()
        self._set_data()
        self._check_dtype()

    @property
    def data(self) -> DataFrame:
        """Data stored in the DataTable as a pandas DataFrame
        """
        return self._data

    @property
    def values(self) -> ndarray:
        """Data stored in the DataTable as a numpy array
        """
        return self._data.values

    @data.setter
    def data(self, data) -> None:
        """Set internal data as a pandas DataFrame, casts array-like objects
        into a DataFrame
        """
        if isinstance(data, DataFrame):
            self._data = data
            return
        # Attempt to cast
        self._data = DataFrame(data)

    def __repr__(self) -> str:
        """Use __repr__ of underlying dataframe
        """
        return self.data.__repr__()

    def __str__(self) -> str:
        """Use __str__ of underlying dataframe
        """
        return self.data.__str__()

    def _repr_html_(self):
        """Allow rendering of underlying DataFrame by default in IPython
        """
        return self.data._repr_html_()

    @abstractmethod
    def _check_shape(
        self,
        expected_rows: int = None,
        expected_columns: int = None
    ) -> None:
        """Check the shape of the data matches the expected parameters. Carry
        out the basic check that a 2x2 array is provided
        """
        if len(self.data.shape) != 2:
            raise ValueError("Data must be a 2D array")

        n_rows, n_cols = self.data.shape
        # allow any value to pass the check if None is specified
        if expected_rows is None:
            expected_rows = n_rows
        if expected_columns is None:
            expected_columns = n_cols

        if expected_rows != n_rows or expected_columns != n_cols:
            raise ValueError(
                f"Expected {expected_rows} x {expected_columns} array. "
                f"Found {n_rows} x {n_cols} array."
            )

    @abstractmethod
    def _init_names(self) -> None:
        """Coerce input names into the format required for DataTable
        """
        pass

    def _auto_name(
        self,
        prefix: str,
        n_names: int,
        alpha: bool,
        start: int = 0
    ) -> List[str]:
        """Automatically generate a list of names to use based on an
        alphabetical A, B, C ... ZY, ZZ or numerical sequence 1, 2, 3, given
        a prefix
        """
        ASCII_A = 65
        result = []
        for n in range(n_names):
            n += start
            if alpha:
                n += start
                seq_0 = chr(ASCII_A + n % 26)
                seq_1 = '' if n // 26 == 0 else chr(ASCII_A + n // 26 - 1)
                seq = seq_1 + seq_0
            else:
                seq = n + 1
            result.append(prefix + seq)

        return result

    def _check_names(self, names: Any, n: int) -> None:
        """Validate that the user defined columns matches the expected format
        """
        if names is None:
            return
        if not isinstance(names, Iterable):
            raise TypeError("Column/row names provided must be an iterable")
        # Check number of column_names provided matches the specified number of
        # groups
        if len(names) != n:
            raise ValueError(
                "The number of columns/rows must match the shape of the data"
            )

    @abstractmethod
    def _set_data(self) -> None:
        """Set the data attribute after checks are complete
        """
        pass

    def _check_dtype(self) -> None:
        """Check the dtype provided in array is correct. In the majority
        of cases, this should be a numerical data type. Attempts to coerce
        dtypes into a valid numerical dtype. 
        """
        for col in self.data.columns:
            self.data[col] = to_numeric(self.data[col])

    @classmethod
    @abstractmethod
    def from_frame(cls, df: DataFrame):
        """Make a DataTable from a DataFrame. Checks that the DataFrame
        provided matches the expected format for that type of DataTable
        """
        pass

    @classmethod
    def _validate_nested_frame(cls, df: DataFrame) -> None:
        """Checks the format of DataFrames which need to be multiindexed or
        "nested" by groupings
        """
        # Check multi-indexed
        if not isinstance(df.columns, MultiIndex):
            raise ValueError(
                f"Columns on a {cls.__name__} must be MultiIndexed"
            )
        # Check correct index levels of 2
        if df.columns.nlevels != 2:
            raise ValueError(
                f"Columns on a {cls.__name__} must be MultiIndexed at Level 2"
            )
        # Check equal sizing of groupings. For XYTables, allow X to be a
        # different size
        slice_from = int(cls.__name__ == "XYTable")
        group_sizes = df \
            .groupby(level=0, axis=1, sort=False) \
            .size() \
            .values[slice_from:]
        # Allow nested tables to have different sized groups but all groups
        # must have at least two subcolumns
        if cls.__name__ == "NestedTable" and not all(group_sizes >= 2):
            raise ValueError("All groups must be at least two subcolumns")
        if not all(group_sizes == group_sizes[0]):
            raise ValueError("Groups must be the same size")
