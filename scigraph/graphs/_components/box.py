from __future__ import annotations

from typing import Self, Never, Any, override, TYPE_CHECKING

from scigraph.graphs.abc import GraphComponent
from scigraph._options import ColumnGraphDirection

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from scigraph.graphs import ColumnGraph


class BoxAndWhiskers(GraphComponent):

    def __init__(
        self,
        kw: dict[str, Any],
        *,
        whis: tuple[float, float] | float = 1.5,  # 1.5 * IQR is the Tukey plot
    ) -> None:
        super().__init__(kw)
        self.whis = whis

    @override
    def draw_xy(self, *args, **kwargs) -> Never:
        raise NotImplementedError

    @override
    def draw_column(self, graph: ColumnGraph, ax: Axes) -> None:
        x = graph.table.values.T
        vert = graph._direction is ColumnGraphDirection.VERTICAL

        for i, id in enumerate(graph.table.dataset_ids):
            props = graph.plot_properties[id]
            bw_kw = props.box_and_whisker_kw()
            bw_kw.update(**self.kw)
            bplot = ax.boxplot(x[i], vert=vert, whis=self.whis, positions=[i], 
                               patch_artist=True, labels=[id], zorder=1, **bw_kw)
            bplot['boxes'][0].set_facecolor(props.barcolor)

    @classmethod
    @override
    def from_opt(cls, opt, kw, **kwargs) -> Self:
        return cls(kw, **kwargs)
