from __future__ import annotations

from typing import Iterator, TYPE_CHECKING, override

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from scigraph.layouts._layout import Layout

if TYPE_CHECKING:
    from scigraph.graphs.abc import Graph


class GridLayout(Layout):

    def __init__(
        self,
        ncols: int,
        nrows: int,
    ) -> None:
        super().__init__()
        self._ncols = ncols
        self._nrows = nrows
        self._graphs: list[list[Graph | None]] = [
            [None for _ in range(ncols)] for _ in range(nrows)
        ]

    @override
    def link_graph(
        self,
        graph: Graph,
        key: tuple[int, int] | None = None
    ) -> None:
        super().link_graph(graph, key)

    @override
    def _draw(self, **fig_kw) -> Figure:
        if "figsize" not in fig_kw:
            figsize = plt.rcParams["figure.figsize"]
            figsize = figsize[0] * self._ncols, figsize[1] * self._nrows
            fig_kw["figsize"] = figsize

        fig, axes = plt.subplots(self._nrows, self._ncols, **fig_kw)
        if isinstance(axes, Axes):  # Only one graph specified
            axes = [[axes]]

        for row, (ax_row, g_row) in enumerate(zip(axes, self._graphs)):
            if isinstance(ax_row, Axes):  # Only one dimension specified
                axes[row] = ax_row = [ax_row]  # type: ignore

            for ax, graph in zip(ax_row, g_row):
                if graph is not None:
                    if self._create_layout_legend:
                        # Suppress axes legends
                        graph.include_legend = False
                    graph.draw(ax)

        return fig

    @override
    def get_position(self, key: tuple[int, int]) -> Graph | None:
        row, col = key
        try:
            return self._graphs[row][col]
        except IndexError as e:
            raise IndexError(f"({row}, {col}) out of range") from e

    @override
    def set_position(self, key: tuple[int, int], graph: Graph) -> None:
        row, col = key
        try:
            self._graphs[row][col] = graph
        except IndexError as e:
            raise IndexError(f"({row}, {col}) out of range") from e

    @override
    def iter_positions(self) -> Iterator[Graph | None]:
        for row in range(self._nrows):
            for col in range(self._ncols):
                yield self._graphs[row][col]

    @override
    def _get_empty_pos_key(self) -> tuple[int, int]:
        for row_idx, row in enumerate(self._graphs):
            for col_idx, g in enumerate(row):
                if g is None:
                    return row_idx, col_idx
        # None found
        raise RuntimeError("Layout full.")
