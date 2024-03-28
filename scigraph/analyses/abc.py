from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

from scigraph.datatables.xy import XYTable
from scigraph.graphs.abc import Graph


class Analysis(ABC):

    @abstractmethod
    def analyze(self) -> None:
        pass


class Plottable(ABC):

    @abstractmethod
    def plot(self, ax: plt.Axes, graph: Graph, *args, **kwargs) -> None:
        pass


class XYAnalysis(Analysis):

    def __init__(self, table: XYTable) -> None:
        self.table = table
