from abc import ABC, abstractmethod


class Graph(ABC):

    @abstractmethod
    def draw(self, ax) -> None:
        pass
