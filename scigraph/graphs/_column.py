from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Self, Literal, override

import matplotlib.pyplot as plt

from scigraph.datatables import ColumnTable
from scigraph.graphs.abc import Graph, TypeChecked
from scigraph.graphs._components.points import Points
from scigraph.graphs._components.errorbars import ErrorBars
from scigraph.graphs._components.connecting_lines import ConnectingLine
from scigraph.graphs._components.axis import ContinuousAxis, CategoricalAxis
from scigraph._options import (
    GraphType,
    ColumnGraphDirection,
    ColumnGraphSubtype
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class ColumnGraph(Graph[ColumnTable]):

    def __init__(
        self,
        table: ColumnTable,
        subtype: Literal["mean", "geometric mean", "median", "individual",
                         "swarm"] | ColumnGraphSubtype,
        direction: Literal["vertical", "horizontal"] | ColumnGraphDirection,
    ) -> None:
        super().__init__()
        # Components
        self._points: Points | None = None
        self._connecting_line: ConnectingLine | None = None
        self._errorbars: ErrorBars | None = None

        # Config
        self._subtype = ColumnGraphSubtype.from_str(subtype)
        self._direction = ColumnGraphDirection.from_str(direction)
        self._init_axis(table.dataset_ids)
        self.link_table(table)

    def _init_axis(self, dataset_ids: list[str]) -> None:
        if self._direction is ColumnGraphDirection.VERTICAL:
            self.xaxis = self.categorical_axis = CategoricalAxis(
                "x", dataset_ids
            )
            self.yaxis = self.continuous_axis = ContinuousAxis("y")
        else:  # Horizontal
            self.xaxis = self.continuous_axis = ContinuousAxis("x")
            self.yaxis = self.categorical_axis = CategoricalAxis(
                "y", dataset_ids
            )

    @override
    def link_table(self, table: ColumnTable) -> None:
        if not isinstance(table, ColumnTable):
            raise TypeError("Only ColumnTables can be linked to ColumnGraphs.")
        self.table = table
        self.categorical_axis.title = table.x_title
        self.continuous_axis.title = table.y_title

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

    @override
    def draw(self, ax: Optional[Axes] = None) -> Axes:
        if ax is None:
            ax = plt.gca()

        self._compile_plot_properties()

        self.xaxis._format_axes(ax)
        self.yaxis._format_axes(ax)

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
    def _checkcode(self) -> TypeChecked.Type:
       return TypeChecked.Type(GraphType.COLUMN, self._subtype)
