"""Contains the ColumnTable class
"""

from typing import Optional, Iterable
from .datatable import DataTable, DataFrame


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

        Parameters
        ----------
        data : array-like
            The data array. If provided in the format of a DataFrame,
            indexes and columns are ignored and only the array is retained. To
            preserve indices and columns, use ColumnTable.from_frame
        n_groups : int
            The number of grouping variables in the dataset
        group_names : Optional[Iterable], optional
            The names of the grouping variables. If none is
            provided, defaults to A, B, C etc, by default None
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
    def from_frame(cls, df: DataFrame):
        """Construct an ColumnTable from a pandas DataFrame while preserving
        the columns and indices. 

        Parameters
        ----------
        df : DataFrame
            DataFrame to be converted into a ColumnTable

        Returns
        -------
        ColumnTable
            The ColumnTable constructed from a DataFrame    
        """
        return cls(df, len(df.columns), df.columns)
