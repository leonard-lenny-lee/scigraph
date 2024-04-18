from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame


class DataTable(ABC):

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
    def as_df(self) -> DataFrame:
        pass

    @property
    @abstractmethod
    def dataset_ids(self) -> list[str]:
        pass

    @abstractmethod
    def get_dataset(self, name: str) -> DataSet:
        pass

    def datasets_itertuples(self) -> Iterator[tuple[str, DataSet]]:
        for id in self.dataset_ids:
            yield id, self.get_dataset(id)

    @classmethod
    def _default_names(cls, n: int, prefix: str = "Group") -> list[str]:
        ascii_a = ord('A')
        names = []
        for i in range(n):
            if i < 26:
                name = f"{prefix} {chr(ascii_a + i)}"
            else:
                k1, k2 = divmod(i, 26)
                name = f"{prefix} {chr(ascii_a + k1)}{chr(ascii_a + k2)}"
            names.append(name)
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


class DataSet(NamedTuple):
    x: NDArray | None
    y: NDArray
