"""Contains the ContigencyTable class
"""

from typing import Optional, Iterable
from ._datatable import DataTable, DataFrame

__all__ = ["ContingencyTable"]


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

        Parameters
        ----------
        data : array-like
            The data array. If provided in the format of a DataFrame,
            indexes and columns are ignored and only the array is retained. To
            preserve indices and columns, use ContigencyTable.from_frame
        n_outcomes : int
            The number of grouping variables in the dataset
        outcome_names : Optional[Iterable], optional
            The names of the grouping variables. If none is provided,
            defaults to A, B, C etc, by default None
        row_names : Optional[Iterable], optional
            The names of the rows,. If None is provided, rows
            will be named 0, 1, 2, 3 ..., by default None
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
    def from_frame(cls, df: DataFrame):
        """Construct an ContigencyTable from a pandas DataFrame while preserving
        the columns and indices. 

        Parameters
        ----------
        df : DataFrame
            DataFrame to be converted into a ContigencyTable

        Returns
        -------
        ContigencyTable
            The ContigencyTable constructed from a DataFrame    
        """
        return cls(df, len(df.columns), df.columns)
