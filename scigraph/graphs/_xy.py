from __future__ import annotations

from typing import override, Literal, Self, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from scigraph.datatables import XYTable
from scigraph.graphs.abc import Graph
from scigraph.graphs._components import (
    Points,
    ErrorBars,
    ConnectingLine,
    ContinuousAxis,
)
from scigraph._options import PointsType, ErrorbarType, ConnectingLineType

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from scigraph.datatables import ColumnTable


class XYGraph(Graph[XYTable]):
    """A graph for x/y data with continuous axes on both dimensions."""

    def __init__(self, table: XYTable) -> None:
        """Create a graph bound to ``table``."""
        super().__init__()

        self.xaxis = ContinuousAxis("x")
        self.yaxis = ContinuousAxis("y")
        self.secondary_yaxis = None
        self._link_table(table)
        self._compile_plot_properties()

    @override
    def _link_table(self, table: XYTable) -> None:
        if not isinstance(table, XYTable):
            raise TypeError("Only XYTables can be linked to XYGraphs.")
        self._table = table
        self.xaxis.title = table.x_title
        self.yaxis.title = table.y_title

    def add_points(
        self,
        ty: Literal["mean", "geometric mean", "median", "individual"],
        **plot_kw,
    ) -> Self:
        """Add a point component and return this graph for chaining."""
        self._register_component(ty, PointsType, Points, plot_kw)
        return self

    def add_errorbars(
        self,
        ty: Literal["sd", "geometric sd", "sem", "ci95", "range"],
        **plot_kw,
    ) -> Self:
        """Add an error-bar component and return this graph for chaining."""
        self._register_component(ty, ErrorbarType, ErrorBars, plot_kw)
        return self

    def add_connecting_line(
        self,
        ty: Literal["mean", "geometric mean", "median", "individual"],
        *,
        join_nan: bool = False,
        **plot_kw,
    ) -> Self:
        """Add a connecting-line component and return this graph for chaining."""
        self._register_component(
            ty, ConnectingLineType, ConnectingLine, plot_kw, join_nan=join_nan
        )
        return self

    def add_area_fill(
        self,
    ) -> Self:
        """Reserved compatibility hook for a future area-fill component."""
        return self

    @override
    def draw(self, ax: Axes | None = None) -> Axes:
        """Draw all registered components and analyses onto ``ax``."""
        if ax is None:
            ax = plt.gca()

        self.xaxis._format_axes(ax)
        self.yaxis._format_axes(ax)

        self._draw_registered(ax, lambda component: component.draw_xy(self, ax))

        return ax

    @classmethod
    def from_column_table(cls, table: ColumnTable) -> Self:
        """Create an XY graph using a column table's row positions as x values."""
        x = np.array(range(table.nrows))
        x = x[:, np.newaxis]
        values = np.hstack((x, table.values))
        xy = XYTable(values, 1, 1, table.ncols)
        xy.x_title = table.x_title
        xy.y_title = table.y_title
        xy.dataset_names = table.dataset_ids
        return cls(xy)
