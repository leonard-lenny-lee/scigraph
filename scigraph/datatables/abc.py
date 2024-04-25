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
        return repr(self.as_df())

    def _sanitize_values(self, values: Any) -> NDArray[np.float64]:
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
    def as_df(self) -> DataFrame: ...

    @property
    @abstractmethod
    def dataset_ids(self) -> list[str]: ...

    @abstractmethod
    def get_dataset(self, name: str) -> DataSet: ...

    def datasets_itertuples(self) -> Iterator[tuple[str, DataSet]]:
        for id in self.dataset_ids:
            yield id, self.get_dataset(id)

    @classmethod
    def _default_names(
        cls,
        n: int,
        prefix: str = "Group",
        numeric_suffix: bool = False
    ) -> list[str]:
        ascii_a = ord('A')
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

    def _verify_names(self, names: list[str], expected_n: int) -> None:
        contains_duplicates = len(names) != len(set(names))
        incorrect_len = len(names) != expected_n

        if contains_duplicates:
            raise ValueError("Duplicate dataset names not allowed.")
        if incorrect_len:
            raise ValueError("Inappropriate number of names provided")

    @classmethod
    def _default_title(cls) -> str:
        cls._instance_count += 1
        return f"{cls.__name__} {cls._instance_count}"


class DataSet(NamedTuple):
    x: NDArray | None
    y: NDArray
