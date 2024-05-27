from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame


class DataTable(ABC):

    _instance_count = 0

    def __repr__(self) -> str:
        """Use pandas DataFrame as the repr, as it's already nicely formatted."""
        return repr(self.as_df())

    def _sanitize_values(self, values: Any) -> NDArray[np.float64]:
        """Helper function used at initialization to check the dimensionality
        of the values array provided and coerce into a uniform np.float64 dtype.
        """
        # Coerce into NDArray[np.float64]
        try:
            if not isinstance(values, np.ndarray):
                values = np.array(values, dtype=float)
            elif values.dtype != np.float64:
                values = values.astype(float)
        except Exception as e:
            raise ValueError(
                "Unable to coerce values into np.float64 ndarray. Check input."
            ) from e
        # Check dimensions
        if values.ndim != 2:
            raise ValueError("Expected 2D matrix.")
        return values

    @abstractmethod
    def as_df(self) -> DataFrame:
        """Represent the DataTable as a pandas DataFrame.

        Wraps the underlying NumPy array in a pandas DataFrame, punting the
        column names and labels into appropriate indices and columns. Allows
        for use of pandas' convienient label-based indexing and slicing
        operations.

        Returns:
            DataFrame representation of the DataTable.
        """

    @property
    @abstractmethod
    def dataset_ids(self) -> list[str]: ...

    @abstractmethod
    def get_dataset(self, name: str) -> DataSet:
        """Access a dataset through its ID.

        Access the raw NumPy arrays of the X and Y values which define
        a dataset.

        Args:
            name: The name of the dataset.

        Returns:
            A dataset object, which is a NamedTuple of the x and y values. x
            may be None for DataTable objects which are not defined by an X
            coordinate.
        """

    def datasets_itertuples(self) -> Iterator[tuple[str, DataSet]]:
        """Iterate through the datasets contained.

        Returns:
            Tuple (dataset_id, DataSet)

        """
        for id in self.dataset_ids:
            yield id, self.get_dataset(id)

    @property
    @abstractmethod
    def values(self) -> NDArray: ...

    @property
    def nrows(self) -> int:
        return self.values.shape[0]

    @property
    def ncols(self) -> int:
        return self.values.shape[1]

    @property
    def title(self) -> str:
        if not hasattr(self, "_title"):
            self._title = self._default_title()
        return self._title

    @title.setter
    def title(self, val: str) -> None:
        self._title = val

    def _get_normalize_values(self) -> NDArray:
        """The slice of values to be targeted by the Normalize analysis"""
        return self.values

    @abstractmethod
    def _set_normalize_values(self, val: NDArray) -> None: ...

    def _verify_names(self, names: list[str], expected_n: int) -> None:
        """General helper function to validate user input. Use in setters of
        important attributes such as dataset_names, which have specific length
        requirements. Raises ValueError if any problems are found.
        """
        contains_duplicates = len(names) != len(set(names))
        incorrect_len = len(names) != expected_n

        if contains_duplicates:
            raise ValueError("Duplicate dataset names not allowed.")
        if incorrect_len:
            raise ValueError("Inappropriate number of names provided")

    @classmethod
    def _default_title(cls) -> str:
        """Generate a unique title in the format {datatable type} {#}"""
        cls._instance_count += 1
        return f"{cls.__name__} {cls._instance_count}"

    @staticmethod
    def _default_names(
        n: int, prefix: str = "Group", numeric_suffix: bool = False
    ) -> list[str]:
        """Helper function to generate a sequence of names. For example:
        [Group A, Group B, ...] or [Row 1, Row 2, ...],
        """
        ascii_a = ord("A")
        names = []
        for i in range(n):
            if numeric_suffix:
                suffix = i + 1
            else:
                if i < 26:
                    suffix = chr(ascii_a + i)
                else:
                    k1, k2 = divmod(i, 26)
                    suffix = chr(ascii_a + k1) + chr(ascii_a + k2)
            names.append(f"{prefix} {suffix}")
        return names

    @classmethod
    @abstractmethod
    def from_dataframe(cls, df: DataFrame, **kwargs) -> DataTable:
        """Construct a DataTable object from a pandas DataFrame."""


class DataSet(NamedTuple):
    x: NDArray | None
    y: NDArray
