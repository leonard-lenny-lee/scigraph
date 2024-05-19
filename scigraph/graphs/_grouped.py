from __future__ import annotations

from typing import Self, Optional, Literal, override, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from scigraph.datatables import GroupedTable
from scigraph.graphs.abc import Graph
from scigraph.graphs._components import (
    CategoricalAxis,
    ContinuousAxis,
    Points,
    ErrorBars,
    ConnectingLine,
    Bars,
    BoxAndWhiskers,
)
from scigraph._options import (
    GroupedGraphDirection,
    GroupedGraphGrouping,
    PointsType,
    ErrorbarType,
    ConnectingLineType,
    BarType,
    LineType,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray


class GroupedGraph(Graph[GroupedTable]):

    def __init__(
        self,
        table: GroupedTable,
        direction: Literal["vertical", "horizontal"],
        grouping: Literal["interleaved", "separated", "stacked"],
    ) -> None:
        super().__init__()

        self._direction = GroupedGraphDirection.from_str(direction)
        self._grouping = GroupedGraphGrouping.from_str(grouping)
        self._init_axis(table._row_names, table._dataset_names)
        self._link_table(table)
        self._compile_plot_properties()

    def _init_axis(self, row_names: list[str], dataset_ids: list[str]) -> None:
        if self._grouping is GroupedGraphGrouping.SEPARATED:
            repeats = len(dataset_ids)
        else:
            repeats = 1
        if self._is_vertical:
            self.categorical_axis = CategoricalAxis("x", row_names, repeats=repeats)
            self.continuous_axis = ContinuousAxis("y")
            self.xaxis, self.yaxis = self.categorical_axis, self.continuous_axis
        else:
            self.continuous_axis = ContinuousAxis("x")
            self.categorical_axis = CategoricalAxis("y", row_names, repeats=repeats)
            self.xaxis, self.yaxis = self.continuous_axis, self.categorical_axis

    @override
    def _link_table(self, table: GroupedTable) -> None:
        if not isinstance(table, GroupedTable):
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
        self, whis: float | tuple[float, float] = 1.5, **plot_kw
    ) -> Self:
        self._register_component("", None, BoxAndWhiskers, plot_kw, whis=whis)
        return self

    @override
    def draw(self, ax: Optional[Axes] = None) -> Axes:
        if ax is None:
            ax = plt.gca()

        for artist in self._components:
            artist.draw_grouped(self, ax)

        for analysis, kws in self._linked_analyses:
            analysis.draw(self, ax, **kws)

        if self.include_legend:
            self._compose_legend(ax)

        self.xaxis._format_axes(ax)
        self.yaxis._format_axes(ax)

        return ax

    @property
    def _is_vertical(self) -> bool:
        return self._direction is GroupedGraphDirection.VERTICAL

    def _x(self) -> NDArray:
        match self._grouping:
            case GroupedGraphGrouping.INTERLEAVED:
                x = self._x_interleaved()
            case GroupedGraphGrouping.SEPARATED:
                x = self._x_separated()
            case GroupedGraphGrouping.STACKED:
                x = self._x_stacked()
        return x

    def _x_interleaved(self) -> NDArray:
        x = np.linspace(0, self.table.nrows - 1, self.table.nrows)
        x_step = 1 / (1 + self.table._n_datasets)
        x_delta = -0.5 + x_step
        out = [x + x_delta + x_step * i for i in range(self.table._n_datasets)]
        return np.vstack(out)

    def _x_separated(self) -> NDArray:
        x = np.linspace(0, self.table.nrows - 1, self.table.nrows)
        out = [x + i * (len(x) + 1) for i in range(self.table._n_datasets)]
        return np.vstack(out)

    def _x_stacked(self) -> NDArray:
        x = np.linspace(0, self.table.nrows - 1, self.table.nrows)
        return np.tile(x, (self.table._n_datasets, 1))
