from __future__ import annotations

from typing import TYPE_CHECKING, Self, Literal, override

import matplotlib.pyplot as plt

from scigraph.datatables import ColumnTable
from scigraph.graphs.abc import Graph
from scigraph.graphs._components.points import Points
from scigraph.graphs._components.errorbars import ErrorBars
from scigraph.graphs._components.connecting_lines import ConnectingLine
from scigraph.graphs._components.axis import Axis

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class ColumnGraph(Graph[ColumnTable]):

    def __init__(
        self,
        table: ColumnTable,
        graph_t: Literal["mean", "geometric mean", "median", "individual",
                         "scatter"],
        direction: Literal["vertical", "horizontal"],
    ) -> None:
        super().__init__()
        # Components
        self._points: Points | None = None
        self._connecting_line: ConnectingLine | None = None
        self._errorbars: ErrorBars | None = None

        # Config
        self._graph_t = graph_t
        self._direction = direction
        self._continuous_axis = Axis()
        self._categorical_axis = Axis()  # TODO - categorical implementation
        
        if direction == "vertical":
            self.xaxis = self._categorical_axis
            self.yaxis = self._continuous_axis
        elif direction == "horizontal":
            self.xaxis = self._continuous_axis
            self.yaxis = self._categorical_axis
        else:
            raise ValueError(f"Invalid direction arg")

        self.link_table(table)

    @override
    def link_table(self, table: ColumnTable) -> None:
        if not isinstance(table, ColumnTable):
            raise TypeError("Only ColumnTables can be linked to ColumnGraphs.")
        self.table = table
        self._categorical_axis.title = table.x_title
        self._continuous_axis.title = table.y_title

    def add_points(self) -> Self:
        if (points := Points.from_str(self._graph_t)) is None:
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

    @override
    def draw(self, ax: Axes | None) -> Axes:
        if ax is None:
            ax = plt.gca()

        self._compile_plot_properties()

        ax.set_xscale(self.xaxis._props.mpl_arg)
        ax.set_yscale(self.yaxis._props.mpl_arg)
        self.xaxis.format_axis(ax.xaxis)
        self.yaxis.format_axis(ax.yaxis)

        ax.set_xlabel(self.xaxis.title)
        ax.set_ylabel(self.yaxis.title)

        if self._points is not None:
            self._points.draw_column(self, ax)

        if self._errorbars is not None:
            self._errorbars.draw_column(self, ax)

        if self._connecting_line is not None:
            self._connecting_line.draw_column(self, ax)

        for analysis in self._linked_analyses:
            analysis.draw(self, ax)

        if self.include_legend:
            self._compose_legend(ax)

        return ax

    @property
    @override
    def _checkcode(self) -> str:
        return self._graph_t
