from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Self, Literal, override

import matplotlib.pyplot as plt

from scigraph.datatables import ColumnTable
from scigraph.graphs.abc import Graph, TypeChecked
from scigraph.graphs._components import Points, ErrorBars, ConnectingLine, Bars
from scigraph.graphs._components.axis import ContinuousAxis, CategoricalAxis
import scigraph._options as sgopt

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from scigraph.graphs.abc import Artist


class ColumnGraph(Graph[ColumnTable]):

    def __init__(
        self,
        table: ColumnTable,
        subtype: Literal["mean", "geometric mean", "median", "individual",
                         "swarm"] | sgopt.ColumnGraphSubtype,
        direction: Literal["vertical", "horizontal"] | sgopt.ColumnGraphDirection,
    ) -> None:
        super().__init__()
        self._artists: list[Artist] = []

        # Config
        self._subtype = sgopt.ColumnGraphSubtype.from_str(subtype)
        self._direction = sgopt.ColumnGraphDirection.from_str(direction)
        self._init_axis(table.dataset_ids)
        self.link_table(table)

    def _init_axis(self, dataset_ids: list[str]) -> None:
        if self._direction is sgopt.ColumnGraphDirection.VERTICAL:
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

    def add_points(
        self,
        ty: Literal["mean", "geometric mean", "median", "individual", "swarm"],
    ) -> Self:
        if (points := Points.from_str(ty)) is None:
            raise ValueError("Invalid plot argument.")

        self._check_component_compatibility(points)
        self._artists.append(points)
        return self

    def add_errorbars(
        self,
        ty: Literal["sd", "geometric sd", "sem", "ci95", "range"],
    ) -> Self:
        if (errorbar := ErrorBars.from_str(ty)) is None:
            raise ValueError("Invalid errorbar argument.")

        self._check_component_compatibility(errorbar)
        self._artists.append(errorbar)
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
        self._artists.append(connecting_line)
        return self

    def add_bars(
        self,
        ty: Literal["mean", "median", "geometric mean"] | sgopt.BarType,
    ) -> Self:
        bar_t = sgopt.BarType.from_str(ty)
        bar = Bars.from_opt(bar_t)
        self._check_component_compatibility(bar)
        self._artists.append(bar)
        return self

    @override
    def draw(self, ax: Optional[Axes] = None) -> Axes:
        if ax is None:
            ax = plt.gca()

        self._compile_plot_properties()

        self.xaxis._format_axes(ax)
        self.yaxis._format_axes(ax)

        for artist in self._artists:
            artist.draw_column(self, ax)

        for analysis in self._linked_analyses:
            analysis.draw(self, ax)

        if self.include_legend:
            self._compose_legend(ax)

        return ax

    @property
    @override
    def _checkcode(self) -> TypeChecked.Type:
       return TypeChecked.Type(sgopt.GraphType.COLUMN, self._subtype)
