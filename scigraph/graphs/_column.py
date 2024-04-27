from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Self, Literal, override

import matplotlib.pyplot as plt
import numpy as np

from scigraph.datatables import XYTable, ColumnTable
from scigraph.graphs.abc import Graph
from scigraph.graphs._components import (
    Points, ErrorBars, ConnectingLine, Bars, BoxAndWhiskers
)
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
        self._compile_plot_properties()

    def _init_axis(self, dataset_ids: list[str]) -> None:
        if self._is_vertical:
            self.categorical_axis = CategoricalAxis("x", dataset_ids)
            self.continuous_axis = ContinuousAxis("y")
            self.xaxis, self.yaxis = self.categorical_axis, self.continuous_axis
        else:
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
        **plot_kw,
    ) -> Self:
        self._register_component(ty, PointsType, Points, plot_kw)
        return self
        
    def add_errorbars(
        self,
        ty: Literal["sd", "geometric sd", "sem", "ci95", "range"],
        **plot_kw,
    ) -> Self:
        self._register_component(ty, ErrorbarType, ErrorBars, plot_kw)
        return self

    def add_connecting_line(
        self,
        ty: Literal["mean", "geometric mean", "median", "individual"],
        *,
        join_nan: bool = False,
        **plot_kw,
    ) -> Self:
        self._register_component(
            ty, ConnectingLineType, ConnectingLine, plot_kw, join_nan=join_nan
        )
        return self
        
    def add_bars(
        self,
        ty: Literal["mean", "median", "geometric mean"],
        **plot_kw,
    ) -> Self:
        self._register_component(ty, BarType, Bars, plot_kw)
        return self

    def add_lines(
        self,
        ty: Literal["mean", "median", "geometric mean"],
        **plot_kw,
    ) -> Self:
        self._register_component(ty, LineType, Bars, plot_kw, line_only=True)
        return self

    def add_box_and_whiskers(
        self,
        whis: float | tuple[float, float] = 1.5,
        **plot_kw
    ) -> Self:
        self._register_component("", None, BoxAndWhiskers, plot_kw, whis=whis)
        return self

    @override
    def draw(self, ax: Optional[Axes] = None) -> Axes:
        if ax is None:
            ax = plt.gca()

        for artist in self._components:
            artist.draw_column(self, ax)

        for analysis, kws in self._linked_analyses:
            analysis.draw(self, ax, **kws)

        if self.include_legend:
            self._compose_legend(ax)

        self.xaxis._format_axes(ax)
        self.yaxis._format_axes(ax)

        return ax

    @property
    def _is_vertical(self) -> bool:
        return self._direction is ColumnGraphDirection.VERTICAL

    @classmethod
    def from_xy_table(
        cls,
        table: XYTable,
        direction: Literal["vertical", "horizontal"],
    ) -> Self:
        # Average replicates and remove X
        values = table.row_statistics("dataset", "mean").analyze().iloc[:, 1:]
        col_table = ColumnTable(np.array(values))
        # Copy values
        col_table.dataset_names = table._dataset_names
        col_table.x_title = table.x_title
        col_table.y_title = table.y_title
        return cls(col_table, direction)
