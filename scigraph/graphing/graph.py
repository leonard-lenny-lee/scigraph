from abc import ABC, abstractmethod

from .. import tables as t


class Graph(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def plot(self, *args, **kwargs) -> None:
        pass
