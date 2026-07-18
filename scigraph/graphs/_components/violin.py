from __future__ import annotations

from typing import Any, Callable, Never, Self, override, TYPE_CHECKING

import numpy as np

from scigraph.graphs.abc import GraphComponent

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from scigraph.graphs import ColumnGraph, GroupedGraph


class ViolinPlot(GraphComponent):
    """A distribution-density violin component for column graphs."""

    def __init__(
        self,
        kw: dict[str, Any],
        *,
        width: float | None = None,
        showmeans: bool = False,
        showmedians: bool = True,
        showextrema: bool = True,
        points: int = 100,
        bw_method: str | float | Callable | None = None,
        body_kws: dict[str, Any] | None = None,
        line_kws: dict[str, Any] | None = None,
    ) -> None:
        """Configure a violin plot.

        Args:
            kw: Additional keyword arguments forwarded to
                :meth:`matplotlib.axes.Axes.violinplot`.
            width: Violin width. Defaults to the column-graph bar width.
            showmeans: Draw mean markers.
            showmedians: Draw median markers.
            showextrema: Draw min/max bars and their central line.
            points: Number of points used to estimate each density.
            bw_method: Kernel-density bandwidth method passed to Matplotlib.
            body_kws: Matplotlib artist properties for filled violin bodies.
            line_kws: Matplotlib artist properties for summary lines.
        """
        super().__init__(kw)
        self.width = width
        self.showmeans = showmeans
        self.showmedians = showmedians
        self.showextrema = showextrema
        self.points = points
        self.bw_method = bw_method
        self.body_kws = {} if body_kws is None else body_kws
        self.line_kws = {} if line_kws is None else line_kws

    @override
    def draw_xy(self, *args, **kwargs) -> Never:
        """Violin plots are not defined for XY graphs."""
        raise NotImplementedError("Violin plots are only supported by ColumnGraph.")

    @override
    def draw_column(self, graph: ColumnGraph, ax: Axes) -> None:
        """Draw one styled violin for each column-table dataset."""
        for position, dataset_id in enumerate(graph.table.dataset_ids):
            values = graph.table.get_dataset(dataset_id).y
            values = values[np.isfinite(values)]
            if values.size < 2:
                # Matplotlib's Gaussian KDE requires at least two values.
                continue

            props = graph.plot_properties[dataset_id]
            width = props.barwidth if self.width is None else self.width
            violin = ax.violinplot(
                values,
                positions=[position],
                vert=graph._is_vertical,
                widths=width,
                showmeans=self.showmeans,
                showmedians=self.showmedians,
                showextrema=self.showextrema,
                points=self.points,
                bw_method=self.bw_method,
                **self.kw,
            )

            body_kws = {
                "facecolor": props.barcolor,
                "edgecolor": props.baredgecolor,
                "linewidth": props.baredgethickness,
            }
            body_kws.update(self.body_kws)
            for body in violin["bodies"]:
                body.set(**body_kws)

            line_kws = {
                "color": props.baredgecolor,
                "linewidth": props.baredgethickness,
                "linestyle": props.ls,
            }
            line_kws.update(self.line_kws)
            for key, artist in violin.items():
                if key != "bodies":
                    artist.set(**line_kws)

    @override
    def draw_grouped(self, *args, **kwargs) -> Never:
        """Grouped violin plots are not yet implemented."""
        raise NotImplementedError("Violin plots are only supported by ColumnGraph.")

    @classmethod
    @override
    def from_opt(cls, opt, kw, **kwargs) -> Self:
        """Create a violin component without an option enum."""
        return cls(kw, **kwargs)
