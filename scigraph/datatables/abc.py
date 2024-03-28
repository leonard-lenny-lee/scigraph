from abc import ABC, abstractmethod

from pandas import DataFrame


class DataTable(ABC):

    @abstractmethod
    def as_df(self) -> DataFrame:
        pass

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
