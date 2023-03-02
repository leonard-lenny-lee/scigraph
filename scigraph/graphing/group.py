
from __future__ import annotations
from typing import List
from warnings import warn

import matplotlib.pyplot as plt

from .graph import Graph
from .cfg import cfg


class Group:

    def __init__(
        self,
        graphs: List[Graph] = None,
        n_rows: int = 1,
        n_cols: int = 1,
        scale: float = 1
    ) -> None:
        if graphs is None:
            self.graphs = []
        else:
            self.graphs = graphs
        self.dimensions = n_rows, n_cols
        self.scale = scale

    def plot(self) -> None:
        n_rows, n_cols = self.dimensions
        width, height = cfg["figure.figsize"]
        figsize = n_cols * width * self.scale, n_rows * height * self.scale
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        [g._plot_axes(ax) for g, ax in zip(self.graphs, axes)]
        exceeds_capacity = self.n_graphs - self.capacity
        if exceeds_capacity > 0:
            warn(
                f"Group has capacity for {self.capacity} while {self.n_graphs} "
                f"are in the group: {exceeds_capacity} graphs have been "
                f"omitted. Expand dimensions to include all graphs",
                RuntimeWarning
            )
        fig.show()

    def add_graph(self, graph: Graph) -> Group:
        self.graphs.append(graph)
        return self

    @property
    def n_graphs(self) -> int: return len(self.graphs)

    @property
    def capacity(self) -> int: return self.dimensions[0] * self.dimensions[1]
