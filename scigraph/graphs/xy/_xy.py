from typing import override, Literal, Self, Any

import matplotlib.pyplot as plt

from ..abc import Graph, GraphTypeCheckComponent
from .._components.points import map_arg_to_points, Points
from .._components.errorbars import map_arg_to_errorbar, ErrorBars
from .._components.axis import Axis

from scigraph.datatables.xy import XYTable
from scigraph._typing import PlottableXYAnalysis

XYGraphType = Literal["mean", "geometric mean", "median", "individual"]


class XYGraph(Graph):

    def __init__(self, graph_t: XYGraphType) -> None:
        self._graph_t = graph_t
        self.table: XYTable = None
        self._points: type[Points] = None
        self._connecting_line = None
        self._errorbars: type[ErrorBars] = None
        self._legend_artists: dict[str, list[Any]] = {}
        self._analyses: list[PlottableXYAnalysis] = []
        self.xaxis = Axis()
        self.yaxis = Axis()

    @override
    def link_table(self, table: XYTable) -> Self:
        if not isinstance(table, XYTable):
            # In the future accept other DataTable types with adapters
            raise TypeError("Only XYTables can be linked to XYGraphs.")
        self.table = table
        self.xaxis.title = table.xname
        self.yaxis.title = table.yname
        self.plot_props = self._init_dataset_plot_props(table.ygroups)
        return self

    def add_points(self) -> Self:
        points = map_arg_to_points(self._graph_t)
        if points is None:
            raise ValueError("Invalid plot argument.")
        self._check_component_compatibility(points)
        self._points = points
        return self

    def add_errorbars(
        self,
        plot: Literal["sd", "geometric sd", "sem", "ci95", "range"],
    ) -> Self:
        errorbar = map_arg_to_errorbar(plot)
        if errorbar is None:
            raise ValueError("Invalid errorbar argument.")
        self._check_component_compatibility(errorbar)
        self._errorbars = errorbar
        return self

    def add_connecting_line(self) -> Self:
        pass

    def add_area_fill(
        self,
    ) -> Self:
        pass

    @override
    def link_analysis(
        self,
        analysis: PlottableXYAnalysis,
    ) -> Self:
        if not isinstance(analysis, PlottableXYAnalysis):
            raise TypeError("Must be a PlottableXYAnalysis")
        if self.table is not None and analysis.table is not self.table:
            raise ValueError("Linked analysis must be from the same DataTable")
        self._analyses.append(analysis)
        return self

    @override
    def draw(self, ax: plt.Axes = None) -> plt.Axes:
        if ax is None:
            ax = plt.gca()
        self.xaxis.format_axis(ax.xaxis)
        self.yaxis.format_axis(ax.yaxis)

        ax.set_xlabel(self.xaxis.title)
        ax.set_ylabel(self.yaxis.title)

        if self.table is None:
            return

        df = self.table.as_df()
        x = df.iloc[:, 0]

        for group in self.table.ygroups:
            props = self.plot_props[group]
            y = df[group].values

            if self._points is not None:
                artist, points = self._points.plot_group(
                    x, y, ax, c=props.color, marker=props.marker)
                self._add_legend_artist(group, artist)
            else:
                # Populate points for other components
                points = map_arg_to_points(self._graph_t)._prepare_xy(x, y)

            if self._errorbars is not None:
                self._errorbars.plot_group(
                    x, y, ax, self.table.xerr, points.y, c=props.color)

        for analysis in self._analyses:
            analysis.plot(ax, self)

        self._compose_legend(ax)
        return ax

    def _compose_legend(self, ax: plt.Axes):
        for k, v in self._legend_artists.items():
            self._legend_artists[k] = v[0] if len(v) == 1 else tuple(v)
        return ax.legend(
            list(self._legend_artists.values()),
            list(self._legend_artists.keys()),
        )

    def _add_legend_artist(self, group: str, artist: Any) -> None:
        if group not in self._legend_artists:
            self._legend_artists[group] = []
        self._legend_artists[group].append(artist)

    def _check_component_compatibility(
        self,
        component: GraphTypeCheckComponent
    ) -> None:
        if self._graph_t not in component.TYPES:
            raise TypeError(
                f"{component.__class__.__name__} is incompatible with"
                f"{self._graph_t} graphs."
            )
