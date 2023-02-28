"""Contains the MultipleVariablesTable class
"""

from enum import Enum
from typing import Optional, List, Dict
from .datatable import DataTable, DataFrame, to_numeric


class VariableType(Enum):

    Continuous = 0,
    Categorical = 1,
    Label = 2


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

        Parameters
        ----------
        data : array-like
            The data array. If provided in the format of a DataFrame,
            indexes and columns are ignored and only the array is retained. To
            preserve indices and columns, use MultipleVariablesTable.from_frame
        n_variables : int
            The number of variables describing each "case"
        variable_names : Optional[List[str]], optional
            The names of the variables describing each case.
            If none is provided, defaults to Variable_A, B, C..., 
            defaults to None
        variable_types : Optional[List[VariableType]], optional
            The types of the variables: Continuous, Categorical or Label
            Variables, provided as a List of VariableType enums.
            If None is provided, variable types will be inferred, with
            numerical types being continuous, while non-numerical types being
            categorical, defaults to None
        """
        self.data = data
        self.n_variables = n_variables
        self.variable_names = variable_names
        self.variable_types = variable_types

        super().__init__()

    def get_variable_types(self) -> Dict[str, VariableType]:
        """Return a dictionary of all the variables contained in the DataTable
        and their VariableTypes

        :return: A dictionary of all variables and variable types
        :rtype: Dict[str, VariableType]
        """
        return dict(zip(self.variable_names, self.variable_types))

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
                    self.data[column] = to_numeric(self.data[column])
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
                self.data[column] = to_numeric(self.data[column])
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
    ):
        """Construct an MultipleVariablesTable from a pandas DataFrame while
        preserving the columns and indices.

        Parameters
        ----------
        df : DataFrame
            DataFrame to be converted into an MultipleVariablesTable
        variable_types : Optional[List[VariableType]], optional
            The type of each variable (column) e.g. Continuous, Categorical or
            Label, specified as a list of VariableType Enumerations. If none
            is provided, variable types will be inferred, by default None

        Returns
        -------
        MultipleVariablesTable
            The MultipleVariablesTable constructed from the DataFrame
        """
        return cls(df.values, len(df.columns), df.columns, variable_types)
