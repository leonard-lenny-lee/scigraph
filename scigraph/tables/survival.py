"""Contains the SurvivalTable class
"""

from typing import Optional, Iterable
from .datatable import DataTable, DataFrame, to_numeric


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

        Parameters
        ----------
        data : array-like
            The data array. If provided in the format of a DataFrame,
            indexes and columns are ignored and only the array is retained. To
            preserve indices and columns, use SurvivalTable.from_frame
        n_groups : int
            The number of grouping (Y) variables in the dataset
        x_name : Optional[str], optional
            The name of the X variable in the dataset, by default None
        group_names : Optional[Iterable], optional
            The names of the grouping (Y) variables, If none is
            provided, defaults to A, B, C etc, by default None
        row_names : Optional[Iterable], optional
            The names of the rows. If none is provided, rows
            will be named 0, 1, 2, 3 ..., by default None
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
        self.data[cols[0]] = to_numeric(self.data[cols[0]])
        for col in cols[1:]:
            self.data[col] = self.data[col].astype("bool")

    @classmethod
    def from_frame(cls, df: DataFrame):
        """Construct an SurvivalTable from a pandas DataFrame while preserving
        the columns and indices. 

        Parameters
        ----------
        df : DataFrame
            DataFrame to be converted into a SurvivalTable

        Returns
        -------
        SurvivalTable
            The SurvivalTable constructed from a DataFrame    
        """
        n_groups = len(df.columns) - 1
        x_name = df.columns[0]
        group_names = df.columns[1:]
        row_names = df.index
        return cls(df, n_groups, x_name, group_names, row_names)
