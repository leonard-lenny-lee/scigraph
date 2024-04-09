from __future__ import annotations

from typing import override, Literal, Self, Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from scigraph.datatables import XYTable
from scigraph.graphs.abc import Graph, TypeChecked
from scigraph.graphs._components.points import Points
from scigraph.graphs._components.errorbars import ErrorBars
from scigraph.graphs._components.connecting_lines import ConnectingLine
from scigraph.graphs._components.axis import Axis
from scigraph.analyses.abc import GraphableAnalysis


class XYGraph(Graph):

    def __init__(
        self,
        table: XYTable,
        graph_t: Literal["mean", "geometric mean", "median", "individual"],
    ) -> None:
        # Components
        self._points: Points | None = None
        self._connecting_line: ConnectingLine | None = None
        self._errorbars: ErrorBars | None = None
        self._legend_artists: dict[str, list[Any]] = {}
        self._linked_analyses: list[GraphableAnalysis] = []

        # Config
        self._graph_t = graph_t
        self.xaxis = Axis()
        self.yaxis = Axis()
        self.secondary_yaxis = None
        self.include_legend = True

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

    def add_area_fill(
        self,
    ) -> Self:
        return self

    @override
    def link_analysis(
        self,
        analysis: GraphableAnalysis,
    ) -> Self:
        if not isinstance(analysis, GraphableAnalysis):
            raise TypeError(f"{type(analysis).__name__} is not graphable.")

        if analysis.table is not self.table:
            raise ValueError("Linked analysis must be from the same DataTable")

        self._linked_analyses.append(analysis)
        return self

    @override
    def draw(self, ax: Axes | None = None) -> Axes:
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

    def _compose_legend(self, ax: Axes):
        labels = []
        handles = []
        for k, v in self._legend_artists.items():
            labels.append(k)
            handles.append(tuple(v))

        return ax.legend(handles, labels)

    def _add_legend_artist(self, group: str, artist: Any) -> None:
        if group not in self._legend_artists:
            self._legend_artists[group] = []

        self._legend_artists[group].append(artist)

    def _check_component_compatibility(
        self,
        component: TypeChecked
    ) -> None:
        if self._graph_t not in component._compatible_types():
            raise TypeError(
                f"{component.__class__.__name__} is incompatible with "
                f"{self._graph_t} graphs."
            )
