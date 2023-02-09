"""Contains the GroupedTable class
"""

from __future__ import annotations

from .datatable import *


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

        Parameters
        ----------
        data : array-like
            The data array. If provided in the format of a DataFrame,
            indexes and columns are ignored and only the array is retained. To
            preserve indices and columns, use GroupedTable.from_frame
        n_groups : int
            The number of groups in the data set
        group_size : int
            The number of data samples per group
        group_names : Optional[Iterable], optional
            The names of the groups. If none is provided, rows
            will be named A, B, C ... ZX, ZY, ZZ, by default None
        row_names : Optional[Iterable], optional
            The names of the rows,. If none is provided, groups
            will be named 0, 1, 2, 3 ..., by default None
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

        Parameters
        ----------
        df : DataFrame
            DataFrame to be converted into a GroupedTable

        Returns
        -------
        GroupedTable
            The GroupedTable constructed from a DataFrame    
        """
        cls._validate_nested_frame(df)
        group_sizes = df.groupby(level=0, axis=1, sort=False).size()
        return cls(len(group_sizes), group_sizes[0], df.columns, df.index)
