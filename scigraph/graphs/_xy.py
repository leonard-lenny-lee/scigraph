from __future__ import annotations

from typing import override, Literal, Self, TYPE_CHECKING

import matplotlib.pyplot as plt

from scigraph.datatables import XYTable
from scigraph.graphs.abc import Graph, TypeChecked
from scigraph.graphs._components.points import Points
from scigraph.graphs._components.errorbars import ErrorBars
from scigraph.graphs._components.connecting_lines import ConnectingLine
from scigraph.graphs._components.axis import ContinuousAxis
from scigraph._options import GraphType, XYGraphSubtype

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class XYGraph(Graph[XYTable]):

    def __init__(
        self,
        table: XYTable,
        subtype: Literal["mean", "geometric mean", "median",
                         "individual"] | XYGraphSubtype,
    ) -> None:
        super().__init__()
        # Components
        self._points: Points | None = None
        self._connecting_line: ConnectingLine | None = None
        self._errorbars: ErrorBars | None = None

        # Config
        self._subtype = XYGraphSubtype.from_str(subtype)
        self.xaxis = ContinuousAxis("x")
        self.yaxis = ContinuousAxis("y")
        self.secondary_yaxis = None

        self.link_table(table)

    @override
    def link_table(self, table: XYTable) -> None:
        if not isinstance(table, XYTable):
            # TODO - Maybe in the future accept other DataTables with adapters?
            raise TypeError("Only XYTables can be linked to XYGraphs.")
        self.table = table
        self.xaxis.title = table.x_title
        self.yaxis.title = table.y_title

    def add_points(self) -> Self:
        if (points := Points.from_str(self._subtype.to_str())) is None:
            raise ValueError("Invalid plot argument.")

        self._check_component_compatibility(points)
        self._points = points
        return self

    def add_errorbars(
        self,
        ty: Literal["sd", "geometric sd", "sem", "ci95", "range"],
    ) -> Self:
        if (errorbar := ErrorBars.from_str(ty)) is None:
            raise ValueError("Invalid errorbar argument.")

        self._check_component_compatibility(errorbar)
        self._errorbars = errorbar
        return self

    def add_connecting_line(
        self,
        ty: Literal["mean", "geometric mean", "median", "individual"],
        *,
        join_nan: bool = False
    ) -> Self:
        if (connecting_line := ConnectingLine.from_str(ty, join_nan)) is None:
            raise ValueError("Invalid connecting line type")

        self._check_component_compatibility(connecting_line)
        self._connecting_line = connecting_line
        return self

    def add_area_fill(
        self,
    ) -> Self:
        return self

    @override
    def draw(self, ax: Axes | None = None) -> Axes:
        if ax is None:
            ax = plt.gca()

        self._compile_plot_properties()

        self.xaxis._format_axes(ax)
        self.yaxis._format_axes(ax)

        if self._points is not None:
            self._points.draw_xy(self, ax)

        if self._errorbars is not None:
            self._errorbars.draw_xy(self, ax)

        if self._connecting_line is not None:
            self._connecting_line.draw_xy(self, ax)

        for analysis in self._linked_analyses:
            analysis.draw(self, ax)

        if self.include_legend:
            self._compose_legend(ax)

        return ax

    @property
    @override
    def _checkcode(self) -> TypeChecked.Type:
        return TypeChecked.Type(GraphType.XY, self._subtype)
