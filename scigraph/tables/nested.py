"""Contains the NestedTable class
"""

from typing import Optional, List, Tuple
from .datatable import DataTable, DataFrame, MultiIndex


class NestedTable(DataTable):

    def __init__(
        self,
        data,
        n_groups: int,
        group_sizes: List[int],
        column_names: Optional[List[Tuple[str, str]]] = None
    ) -> None:
        """A nested table is used when there are two levels of nested or
        hierarchical replication.

        Parameters
        ----------
        data : array-like
            The data array. If provided in the format of a DataFrame,
            indexes and columns are ignored and only the array is retained. To
            preserve indices and columns, use NestedTable.from_frame
        n_groups : int
            The number of highest-level groupings in the dataset
        group_sizes : List[int]
            The numbers of sub-groups within each group. Each group must have
            at least two sub-groups.
        column_names : Optional[List[Tuple[str, str]]], optional
            The names of the groups and subgroups, provided as a list of
            tuples, by default None
        """
        self.data = data
        self.n_groups = n_groups
        self.group_sizes = group_sizes
        self.column_names = column_names

        super().__init__()

    def _check_shape(self) -> None:
        """Check the number of groups matches the number of group sizes
        and the number of columns matches the group sizes.
        """
        if self.n_groups != len(self.group_sizes):
            raise ValueError(
                f"{self.n_groups} specified, {len(self.group_sizes)} group_sizes provided."
            )
        expected_cols = sum(self.group_sizes)
        super()._check_shape(expected_columns=expected_cols)

    def _init_names(self) -> None:
        """Generate the nested column names if none are provided.
        """
        if self.column_names is None:
            group_names = self._auto_name(
                prefix="Group_", n_names=self.n_groups, alpha=True
            )
            self.column_names = []
            subcol_idx = 0
            for group, size in zip(group_names, self.group_sizes):
                self.column_names.extend(
                    [(group, subcol_idx + n) for n in range(size)])
                subcol_idx += size
            return
        self._check_names(self.column_names, sum(self.group_sizes))
        if not all(map(lambda name: isinstance(name, tuple) and len(name) == 2, self.column_names)):
            raise ValueError(
                "\"column_names\" must be nested to a level of 2."
            )

    def _set_data(self) -> None:
        """Generate MultiIndex columns from tuples are set DataFrame
        """
        columns = MultiIndex.from_tuples(self.column_names)
        self.data = DataFrame(self.data.values, columns=columns)

    @classmethod
    def from_frame(cls, df: DataFrame):
        """Construct an NestedTable from a pandas DataFrame while preserving
        the columns and indices. 

        Parameters
        ----------
        df : DataFrame
            DataFrame to be converted into a NestedTable

        Returns
        -------
        NestedTable
            The NestedTable constructed from a DataFrame    
        """
        super()._validate_nested_frame(df)
        group_sizes = list(df.groupby(level=0, axis=1, sort=False).size())
        return cls(df.values, len(group_sizes), group_sizes, df.columns)
