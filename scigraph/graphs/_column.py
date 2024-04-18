from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Self, Literal, override

import matplotlib.pyplot as plt

from scigraph.datatables import ColumnTable
from scigraph.graphs.abc import Graph
from scigraph.graphs._components import Points, ErrorBars, ConnectingLine, Bars
from scigraph.graphs._components.axis import ContinuousAxis, CategoricalAxis
from scigraph._options import (
    ColumnGraphDirection, PointsType, ConnectingLineType, ErrorbarType,
    BarType, LineType
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class ColumnGraph(Graph[ColumnTable]):

    def __init__(
        self,
        table: ColumnTable,
        direction: Literal["vertical", "horizontal"],
    ) -> None:
        super().__init__()

        self._direction = ColumnGraphDirection.from_str(direction)
        self._init_axis(table.dataset_ids)
        self._link_table(table)

    def _init_axis(self, dataset_ids: list[str]) -> None:
        if self._direction is ColumnGraphDirection.VERTICAL:
            self.categorical_axis = CategoricalAxis("x", dataset_ids)
            self.continuous_axis = ContinuousAxis("y")
            self.xaxis, self.yaxis = self.categorical_axis, self.continuous_axis
        else:  # Horizontal
            self.continuous_axis = ContinuousAxis("x")
            self.categorical_axis = CategoricalAxis("y", dataset_ids)
            self.xaxis, self.yaxis = self.continuous_axis, self.categorical_axis

    @override
    def _link_table(self, table: ColumnTable) -> None:
        if not isinstance(table, ColumnTable):
            raise TypeError("Only ColumnTables can be linked to ColumnGraphs.")

        self._table = table
        self.categorical_axis.title = table.x_title
        self.continuous_axis.title = table.y_title

    def add_points(
        self,
        ty: Literal["mean", "geometric mean", "median", "individual", "swarm"],
    ) -> Self:
        self._register_component(ty, PointsType, Points)
        return self
        
    def add_errorbars(
        self,
        ty: Literal["sd", "geometric sd", "sem", "ci95", "range"],
    ) -> Self:
        self._register_component(ty, ErrorbarType, ErrorBars)
        return self

    def add_connecting_line(
        self,
        ty: Literal["mean", "geometric mean", "median", "individual"],
        *,
        join_nan: bool = False
    ) -> Self:
        self._register_component(
            ty, ConnectingLineType, ConnectingLine, join_nan=join_nan
        )
        return self
        
    def add_bars(
        self,
        ty: Literal["mean", "median", "geometric mean"]
    ) -> Self:
        self._register_component(ty, BarType, Bars)
        return self

    def add_lines(
        self,
        ty: Literal["mean", "median", "geometric mean"]
    ) -> Self:
        self._register_component(ty, LineType, Bars, line_only=True)
        return self

    @override
    def draw(self, ax: Optional[Axes] = None) -> Axes:
        if ax is None:
            ax = plt.gca()

        self._compile_plot_properties()

        self.xaxis._format_axes(ax)
        self.yaxis._format_axes(ax)

        for artist in self._components:
            artist.draw_column(self, ax)

        for analysis in self._linked_analyses:
            analysis.draw(self, ax)

        if self.include_legend:
            self._compose_legend(ax)

        return ax
