from __future__ import annotations

from typing import override

from numpy.typing import NDArray
from pandas import DataFrame, Index

from .abc import DataSet, DataTable
from scigraph.config import SG_DEFAULTS


class ColumnTable(DataTable):
    
    def __init__(self, values: NDArray) -> None:
        # Check array shape
        if values.ndim != 2:
            raise ValueError("Expected 2D matrix.")
        _, n_cols = values.shape

        # Protected attributes
        self._values = values
        self._n_datasets = n_cols
        self._dataset_names = self._default_names(n_cols)

        self.x_title: str = SG_DEFAULTS["datatables.column.x_title"]
        self.y_title: str = SG_DEFAULTS["datatables.column.y_title"]

    @override
    def as_df(self) -> DataFrame:
        return DataFrame(self._values, columns=Index(self._dataset_names))

    @property
    @override
    def dataset_ids(self) -> list[str]:
        return self._dataset_names

    @override
    def get_dataset(self, name: str) -> DataSet:
        try:
            idx = self._dataset_names.index(name)
        except ValueError as e:
            raise ValueError(f"{name} is not a dataset name.") from e
        return DataSet(
            x=None,
            y=self._values[:, idx]
        )

    @property
    def values(self) -> NDArray:
        return self._values

    @property
    def n_datasets(self) -> int:
        return self._n_datasets

    @property
    def dataset_names(self) -> set[str]:
        return set(self._dataset_names)

    @dataset_names.setter
    def dataset_names(self, names: list[str]) -> None:
        contains_duplicates = len(names) != len(set(names))
        incorrect_len = len(names) != self._n_datasets

        if contains_duplicates:
            raise ValueError("Duplicate dataset names not allowed.")
        if incorrect_len:
            raise ValueError("Inappropriate number of names provided")

        self._dataset_names = names
