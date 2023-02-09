"""Contains the PartsOfWholeTable class
"""

from __future__ import annotations

from .datatable import *


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

        Parameters
        ----------
        data : array-like
            The data array. If provided in the format of a DataFrame,
            indexes and columns are ignored and only the array is retained. To
            preserve indices and columns, use PartsOfWholeTable.from_frame
        n_columns : int
            The number of columns
        column_names : Optional[List[str]], optional
            The names of the columns, by default None
        row_names : Optional[List[str]], optional
            The names of the rows, by default None
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
        """Automatically generate column names if none if provided"""
        if self.column_names is None:
            self.column_names = self._auto_name(
                prefix="", n_names=self.n_columns, alpha=True
            )
            return
        self._check_names(self.column_names, self.n_columns)

    def _set_data(self) -> None:
        """Set the DataFrame using the generated column and row names"""
        self.data = DataFrame(
            self.data.values, self.row_names, self.column_names
        )

    @classmethod
    def from_frame(cls, df: DataFrame) -> PartsOfWholeTable:
        """Construct an PartsOfWholeTable from a pandas DataFrame while
        preserving the columns and indices. 

        Parameters
        ----------
        df : DataFrame
            DataFrame to be converted into a PartsOfWholeTable

        Returns
        -------
        PartsOfWholeTable
            The PartsOfWholeTable constructed from a DataFrame    
        """
        return cls(len(df.columns), df.columns, df.index)
