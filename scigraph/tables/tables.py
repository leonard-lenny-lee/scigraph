"""Contains the DataTable classes upon which are used by the graphing and
analysis modules. The type of DataTable object will determine which types of
analyses and graphs can be constructed from the dataset. These are clones of
the DataTable types in GraphPad Prism 9. See the user guide for in-depth usage
https://www.graphpad.com/guides/prism/latest/user-guide/index.htm

All DataTables are thin wrappers for pandas DataFrames. While the functions and
methods in the graphing and analysis modules will operate on pandas DataFrames,
DataTables will validate the data structure and shape and guarantee the
correct operation of graphing and analysis tools.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from typing import Optional, List, Any

import pandas as pd
import numpy as np

from pandas import DataFrame, MultiIndex
from numpy import ndarray


class VariableType(Enum):
    Continuous = 0,
    Categorical = 1,
    Label = 2


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
                f"Expected {expected_rows} x {expected_columns} array. Found {n_rows} x {n_cols} array."
            )

    @abstractmethod
    def _init_names(self) -> None:
        """Coerce input names into the format required for DataTable
        """
        pass

    def _auto_name(self, prefix: str, n_names: int, alpha: bool) -> List[str]:
        """Automatically generate a list of names to use based on an
        alphabetical A, B, C ... ZY, ZZ or numerical sequence 1, 2, 3, given
        a prefix
        """
        ASCII_A = 65

        result = []
        for n in range(n_names):
            if alpha:
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
            self.data[col] = pd.to_numeric(self.data[col])

    @classmethod
    @abstractmethod
    def from_frame(cls, df: DataFrame) -> DataTable:
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
        group_sizes = df.groupby(
            level=0, axis=1, sort=False).size().values[slice_from:]
        if not all(group_sizes == group_sizes[0]):
            raise ValueError("Groups must be the same size")


class XYTable(DataTable):

    def __init__(
        self,
        data,
        n_x_columns: int,
        n_y_groups: int,
        y_group_size: int,
        x_name: Optional[str] = None,
        row_names: Optional[Iterable] = None,
        group_names: Optional[Iterable] = None
    ) -> None:
        """An XY table is a graph where every point is defined by both an X
        and a Y value. This kind of data are often fit with linear or nonlinear
        regression.

        :param data: An array of the data, including X and y. If provided in a
            DataFrame, the indices are ignored and only the array is used. To
            transform a DataFrame directly into the XYTable, use the method
            XY.to_frame(df)
        :type data: array-like
        :param n_x_columns: The number of columns which represent the X data.
            X data is always assumed to be aligned left
        :type n_x_columns: int
        :param n_y_groups: The number of Y groups in the dataset
        :type n_y_groups: int
        :param y_group_size: The number of columns per Y group
        :type y_group_size: int
        :param x_name: The name of the X columns, defaults to None
        :type x_name: Optional[str], optional
        :param row_names: The names of the rows. If none is provided, rows
            will be named 0, 1, 2, 3..., defaults to None
        :type row_names: Optional[Iterable], optional
        :param group_names: The names of the Y groups. If none is provided,
            groups will be named A, B, C ... ZX, ZY, ZZ, defaults to None
        :type group_names: Optional[Iterable], optional
        """
        self.data = data
        self.n_x_columns = n_x_columns
        self.n_y_groups = n_y_groups
        self.y_group_size = y_group_size
        self.x_name = x_name
        self.row_names = row_names
        self.group_names = group_names

        super().__init__()

    def _check_shape(self) -> None:
        """Check that there is a column for each replicate of each y group and
        the replicates for the X group
        """
        expected_cols = self.n_y_groups * self.y_group_size + self.n_x_columns
        super()._check_shape(expected_columns=expected_cols)

    def _init_names(self) -> None:
        """Create the necessary column_names for indexing the DataTable from the
        specified parameters. If no column_names are provided generate a default
        of Group A, B, C, ... ZY, ZZ
        """
        # Generate x_names
        self.x_name = "X" if self.x_name is None else f"X: {self.x_name}"
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
        X_col = [(self.x_name, n) for n in range(1, self.n_x_columns + 1)]
        y_col = [(group, n) for group in self.group_names
                 for n in range(1, self.y_group_size + 1)]
        columns = MultiIndex.from_tuples(X_col + y_col, names=["Group", "n"])

        self.data = DataFrame(
            self.data.values, index=self.row_names, columns=columns
        )

    @classmethod
    def from_frame(cls, df: DataFrame) -> XYTable:
        """Construct an XYTable from a pandas DataFrame while preserving the
        columns and indices.

        :param df: DataFrame to be converted into an XYTable
        :type df: DataFrame
        :return: The XYTable constructed from the DataFrame
        :rtype: XYTable
        """
        cls._validate_nested_frame(df)
        # Construct arguments for instance initialisation
        group_sizes = df.groupby(level=0, axis=1, sort=False).size()
        data = df.values
        n_x_columns = group_sizes[0]
        n_y_groups = len(group_sizes[1:])
        group_size = group_sizes[1]
        row_names = df.index
        x_name, column_names = group_sizes.index[0], list(
            group_sizes.index[1:])
        return cls(data, n_x_columns, n_y_groups, group_size, x_name, row_names, column_names)


class ColumnTable(DataTable):

    def __init__(
        self,
        data,
        n_groups: int,
        group_names: Optional[Iterable] = None
    ) -> None:
        """Use column tables if your groups are defined by one grouping
        variable, perhaps control vs. treated, or placebo vs. low-dose vs.
        high-dose. Each column defines one group within the same grouping
        variable.

        :param data: The data array. If provided in the format of a DataFrame,
            indexes and columns are ignored and only the array is retained. To
            preserve indices and columns, use ColumnTable.from_frame
        :type data: array-like
        :param n_groups: The number of grouping variables in the dataset
        :type n_groups: int
        :param group_names: The names of the grouping variables. If none is
            provided, defaults to A, B, C etc, defaults to None
        :type group_names: Optional[Iterable], optional
        """
        self.data = data
        self.n_groups = n_groups
        self.group_names = group_names
        super().__init__()

    def _check_shape(self) -> None:
        """Check that there is one column per group
        """
        super()._check_shape(expected_columns=self.n_groups)

    def _init_names(self):
        """Check that the number of column names matched the number of columns
        in data array. Automatically generate names if none are provided
        """
        if self.group_names is None:
            self.group_names = self._auto_name(
                prefix="Group_", n_names=self.n_groups, alpha=True
            )
            return
        self._check_names(self.group_names, self.n_groups)

    def _set_data(self) -> None:
        """Create a DataFrame from generated columns
        """
        self.data = DataFrame(self.data.values, columns=self.group_names)

    @classmethod
    def from_frame(cls, df: DataFrame) -> ColumnTable:
        """Construct a ColumnTable from a pandas DataFrame while preserving the
        columns and indices.

        :param df: DataFrame to be converted into a ColumnTable
        :type df: DataFrame
        :return: The ColumnTable constructed from the DataFrame
        :rtype: ColumnTable
        """
        return cls(df, len(df.columns), df.columns)


class GroupedTable(DataTable):

    def __init__(
        self,
        data,
        n_groups: int,
        group_size: int,
        group_names: Optional[Iterable] = None,
        row_names: Optional[Iterable] = None
    ):
        """Grouped data tables are similar to Column data tables, but are 
        designed for two grouping variables. Groups (or levels) of one 
        grouping variable are defined by rows; the groups (levels) of the other
        grouping variable are defined by columns.

        :param data: The data array. If provided in the format of a DataFrame,
            indexes and columns are ignored and only the array is retained. To
            preserve indices and columns, use GroupedTable.from_frame
        :type data: array-like
        :param n_groups: The number of groups in the data set
        :type n_groups: int
        :param group_size: The number of data samples per group
        :type group_size: int
        :param group_names: The names of the groups. If none is provided, rows
            will be named A, B, C ... ZX, ZY, ZZ, defaults to None
        :type group_names: Optional[Iterable], optional
        :param row_names: The names of the rows,. If none is provided, groups
            will be named 0, 1, 2, 3 ..., defaults to None
        :type row_names: Optional[Iterable], optional
        """
        self.data = data
        self.n_groups = n_groups
        self.group_size = group_size
        self.group_names = group_names
        self.row_names = row_names

        super().__init__()

    def _check_shape(self) -> None:
        """Check there are columns for each of the replicates for each group
        """
        expected_cols = self.n_groups * self.group_size
        super()._check_shape(expected_columns=expected_cols)

    def _init_names(self) -> None:
        """Create the necessary group names for level 0 multiindex of the
        DataTable.
        """
        if self.group_names is None:
            self.group_names = self._auto_name(
                prefix="Group_", n_names=self.n_groups, alpha=True
            )
            return
        n_rows, _ = self.data.shape
        self._check_names(self.group_names, self.n_groups)
        self._check_names(self.row_names, n_rows)

    def _set_data(self) -> None:
        """Construct the MultiIndex for the DataFrame and set attribute
        """
        columns = [(group, n) for group in self.group_names
                   for n in range(1, self.group_size + 1)]
        columns = MultiIndex.from_tuples(columns)
        self.data = DataFrame(
            self.data.values, index=self.row_names, columns=columns
        )

    @classmethod
    def from_frame(cls, df: DataFrame) -> GroupedTable:
        """Construct an GroupedTable from a pandas DataFrame while preserving
        the columns and indices.

        :param df: DataFrame to be converted into an GroupedTable
        :type df: DataFrame
        :return: The GroupedTable constructed from the DataFrame
        :rtype: GroupedTable
        """
        cls._validate_nested_frame(df)
        group_sizes = df.groupby(level=0, axis=1, sort=False).size()
        return cls(len(group_sizes), group_sizes[0], df.columns, df.index)


class ContingencyTable(DataTable):

    def __init__(
        self,
        data,
        n_outcomes: int,
        outcome_names: Optional[Iterable] = None,
        row_names: Optional[Iterable] = None
    ) -> None:
        """Contingency tables - like Grouped data tables - are also designed
        for data described by two grouping variables. However, these tables
        are used to tabulate the actual number of subjects (or observations) 
        that belong to each of the groups defined by the rows and columns

        :param data: The data array. If provided in the format of a DataFrame,
            indexes and columns are ignored and only the array is retained. To
            preserve indices and columns, use ContigencyTable.from_frame
        :type data: array-like
        :param n_outcomes: The number of grouping variables in the dataset
        :type n_outcomes: int
        :param outcome_names: The names of the grouping variables. If none is
            provided, defaults to A, B, C etc, defaults to None
        :type outcome_names: Optional[Iterable], optional
        :param row_names: The names of the rows,. If none is provided, rows
            will be named 0, 1, 2, 3 ..., defaults to None
        :type row_names: Optional[Iterable], optional
        """
        self.data = data
        self.n_outcomes = n_outcomes
        self.outcome_names = outcome_names
        self.row_names = row_names

        super().__init__()

    def _check_shape(self) -> None:
        """Check there is one column for each group
        """
        super()._check_shape(expected_columns=self.n_outcomes)

    def _init_names(self) -> None:
        """Create the necessary outcome names, if none is provided. Otherwise,
        validate the names
        """
        if self.outcome_names is None:
            self.outcome_names = self._auto_name(
                prefix="Outcome_", n_names=self.n_outcomes, alpha=True
            )
            return
        n_rows, n_cols = self.data.shape
        self._check_names(self.outcome_names, n_cols)
        self._check_names(self.row_names, n_rows)

    def _set_data(self) -> None:
        """Set the column and indices with initialised outcome and row names
        """
        self.data = DataFrame(
            self.data.values, index=self.row_names, columns=self.outcome_names
        )

    @classmethod
    def from_frame(cls, df: DataFrame) -> ContingencyTable:
        """Construct a ContigencyTable from a pandas DataFrame while preserving
        the columns and indices.

        :param df: DataFrame to be converted into an ContigencyTable
        :type df: DataFrame
        :return: The ContigencyTable constructed from the DataFrame
        :rtype: ContigencyTable
        """
        return cls(df, len(df.columns), df.columns)


class SurvivalTable(DataTable):

    def __init__(
        self,
        data,
        n_groups: int,
        x_name: Optional[str] = None,
        group_names: Optional[Iterable] = None,
        row_names: Optional[Iterable] = None
    ) -> None:
        """Survival tables are used to perform survival analysis using the
        Kaplan-Meier method. Each row represents a different subject or
        individual. The X column is used to enter elapsed survival time, while
        the Y columns are used to enter the outcome (event or censored) for
        different groups of a single grouping variable.

        :param data: The data array. If provided in the format of a DataFrame,
            indexes and columns are ignored and only the array is retained. To
            preserve indices and columns, use Survival.from_frame
        :type data: array-like
        :param n_groups: The number of grouping (Y) variables in the dataset
        :type n_groups: int
        :param x_name: The name of the X variable in the dataset,
            defaults to None
        :type x_name: Optional[str], optional
        :param group_names: The names of the grouping (Y) variables, If none is
            provided, defaults to A, B, C etc, defaults to None
        :type group_names: Optional[Iterable], optional
        :param row_names: The names of the rows,  If none is provided, rows
            will be named 0, 1, 2, 3 ..., defaults to None
        :type row_names: Optional[Iterable], optional
        """
        self.data = data
        self.n_groups = n_groups
        self.x_name = x_name
        self.group_names = group_names
        self.row_names = row_names

        super().__init__()

    def _check_shape(self) -> None:
        """Check that there is one column per group, plus one for X data
        """
        expected_columns = self.n_groups + 1
        super()._check_shape(expected_columns=expected_columns)

    def _init_names(self) -> None:
        """If none is provided, create group names, otherwise validate
        """
        self.x_name = "X" if self.x_name is None else f"X: {self.x_name}"
        if self.group_names is None:
            self.group_names = self._auto_name(
                prefix="Group_", n_names=self.n_groups, alpha=True
            )
            return
        n_rows, _ = self.data.shape
        self._check_names(self.group_names, self.n_groups)
        self._check_names(self.row_names, n_rows)

    def _set_data(self) -> None:
        """Construct column names from X and Y names, set DataFrame indices
        and columns
        """
        columns = [self.x_name].extend(self.group_names)
        self.data = DataFrame(self.data.values, self.row_names, columns)

    def _check_dtype(self) -> None:
        """Check that the X column is numeric, while the rest are boolean
        """
        cols = self.data.columns
        self.data[cols[0]] = pd.to_numeric(self.data[cols[0]])
        for col in cols[1:]:
            self.data[col] = self.data[col].astype("bool")

    @classmethod
    def from_frame(cls, df: DataFrame) -> SurvivalTable:
        """Construct an SurvivalTable from a pandas DataFrame while preserving
        the columns and indices.

        :param df: DataFrame to be converted into an SurvivalTable
        :type df: DataFrame
        :return: The SurvivalTable constructed from the DataFrame
        :rtype: SurvivalTable
        """
        n_groups = len(df.columns) - 1
        x_name = df.columns[0]
        group_names = df.columns[1:]
        row_names = df.index
        return cls(df, n_groups, x_name, group_names, row_names)


class PartsOfWholeTable(DataTable):

    def __init__(
        self,
        data,
        n_columns: int,
        column_names: Optional[List[str]] = None,
        row_names: Optional[List[str]] = None,
    ):
        """A Parts of whole table is used when it makes sense to ask: What
        fraction of the total is each value? This table is often used to make
        pie charts

        :param data: The data array. If provided in the format of a DataFrame,
            indexes and columns are ignored and only the array is retained. To
            preserve indices and columns, use PartsOfWholeTable.from_frame
        :type data: array-like
        :param n_columns: The number of columns
        :type n_columns: int
        :param column_names: The names of the columns, defaults to None
        :type column_names: Optional[List[str]], optional
        :param row_names: The names of the rows, defaults to None
        :type row_names: Optional[List[str]], optional
        """
        self.data = data
        self.n_columns = n_columns
        self.column_names = column_names
        self.row_names = row_names

        super().__init__()

    def _check_shape(self) -> None:
        """Check number of columns in array matches with number specified
        """
        super()._check_shape(expected_columns=self.n_columns)

    def _init_names(self) -> None:
        if self.column_names is None:
            self.column_names = self._auto_name(
                prefix="", n_names=self.n_columns, alpha=True
            )
            return
        self._check_names(self.column_names, self.n_columns)

    def _set_data(self) -> None:
        self.data = DataFrame(
            self.data.values, self.row_names, self.column_names
        )

    @classmethod
    def from_frame(cls, df: DataFrame) -> PartsOfWholeTable:
        return cls(len(df.columns), df.columns, df.index)


class MultipleVariablesTable(DataTable):

    def __init__(
        self,
        data,
        n_variables: int,
        variable_names: Optional[List[str]] = None,
        variable_types: Optional[List[VariableType]] = None,
    ) -> None:
        """A multiple variables data table is arranged the same way most
        statistics programs organize data. Each row is a different observation
        or "case" (experiment, animal, etc.). Each column is a different
        variable. Additionally, variables can be identified as either a
        continuous variable, a categorical variable, or a label variable,
        while values for categorical and label variables can be entered as
        text ("Female" and "Male" in the example below).

        :param data: The data array. If provided in the format of a DataFrame,
            indexes and columns are ignored and only the array is retained. To
            preserve indices and columns, use MultipleVariablesTable.from_frame
        :type data: array-like
        :param n_variables: The number of variables describing each case
        :type n_variables: int
        :param variable_names: The names of the variables describing each case.
            If none is provided, defaults to Variable_A, B, C..., 
            defaults to None
        :type variable_names: Optional[List[str]], optional
        :param variable_types: The types of the variables: Continuous,
            Categorical or Label Variables, provided as a List of VariableType
            enums. If None is provided, variable types will be inferred, with
            numerical types being continuous, while non-numerical types being
            categorical, defaults to None
        :type variable_types: Optional[List[VariableType]], optional
        """
        self.data = data
        self.n_variables = n_variables
        self.variable_names = variable_names
        self.variable_types = variable_types

        super().__init__()

    def _check_shape(self) -> None:
        """Check the number of variables specified matches the number of
        columns in the data array
        """
        expected_columns = self.n_variables
        super()._check_shape(expected_columns=expected_columns)

    def _init_names(self) -> None:
        """Check the number of variable names provided matches the number of
        columns in the data array. Automatically generate variable names if
        None are provided
        """
        if self.variable_names is None:
            self.variable_names = self._auto_name(
                prefix="Variable_", n_names=self.n_variables, alpha=True
            )
            return
        self._check_names(self.variable_names, self.n_variables)

    def _set_data(self) -> None:
        """Set the DataFrame using variable_names as columns
        """
        self.data = DataFrame(self.data.values, columns=self.variable_names)

    def _check_dtype(self) -> None:
        """If the variable types are specified, coerce the Series dtypes into
        the specified types. If not, infer and then coerce
        """
        # Infer variable types if no variable types are provided
        if self.variable_types is None:
            variable_types = []
            for column in self.data.columns:
                try:
                    self.data[column] = pd.to_numeric(self.data[column])
                    variable_types.append(VariableType.Continuous)
                except ValueError:
                    self.data[column] = self.data[column].astype("category")
                    variable_types.append(VariableType.Categorical)
            self.variable_types = variable_types
            return

        # If variable types are provided, validate and coerce
        if len(self.variable_types) != self.n_variables:
            raise ValueError("Each variable must be assigned a single type")

        for var_type, column in zip(self.variable_types, self.data.columns):
            if not isinstance(var_type, VariableType):
                raise ValueError("variable_types must be list of VariableType")

            if var_type == VariableType.Continuous:
                self.data[column] = pd.to_numeric(self.data[column])
            elif var_type == VariableType.Categorical:
                self.data[column] = self.data[column].astype("category")
            else:
                assert var_type == VariableType.Label
                self.data[column] = self.data[column].astype("object")

    @classmethod
    def from_frame(
        cls,
        df: DataFrame,
        variable_types: Optional[List[VariableType]] = None
    ) -> MultipleVariablesTable:
        """Construct an MultipleVariablesTable from a pandas DataFrame while
        preserving the columns and indices.

        :param df: DataFrame to be converted into an MultipleVariablesTable
        :type df: DataFrame
        :return: The MultipleVariablesTable constructed from the DataFrame
        :rtype: MultipleVariablesTable
        """
        return cls(df.values, len(df.columns), df.columns, variable_types)


class NestedTable(DataTable):

    pass
