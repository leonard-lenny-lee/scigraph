from typing import Protocol, runtime_checkable

from scigraph.datatables.xy import XYTable


@runtime_checkable
class Analysis(Protocol):
    def analyze(self): ...


@runtime_checkable
class Plottable(Protocol):
    def plot(self, ax, graph): ...


@runtime_checkable
class XY(Protocol):
    table: XYTable


@runtime_checkable
class PlottableXYAnalysis(XY, Plottable, Analysis, Protocol):
    ...
