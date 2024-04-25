from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import cycle
from typing import Any, Optional, Self, TYPE_CHECKING

from scigraph.analyses.abc import GraphableAnalysis
from scigraph.config import SG_DEFAULTS
from scigraph.graphs._props import PlotProps

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from scigraph.datatables.abc import DataTable
    from scigraph.graphs import XYGraph, ColumnGraph
    from scigraph._options import Option


class Graph[T: DataTable](ABC):

    def __init__(self) -> None:
        self._legend_artists: dict[str, list[Any]] = {}
        self._linked_analyses: list[GraphableAnalysis] = []
        self._components: list[GraphComponent] = []
        self.include_legend = True

    @abstractmethod
    def _link_table(self, table: T) -> None:
        self._table: T
        ...

    @property
    def table(self) -> T:
        return self._table

    @property
    def title(self) -> str:
        if not hasattr(self, "_title"):
            self._title = self.table.title
        return self._title

    @title.setter
    def title(self, val: str) -> None:
        self._title = val

    def link_analysis(self, analysis: GraphableAnalysis) -> Self:
        if not isinstance(analysis, GraphableAnalysis):
            raise TypeError(f"{type(analysis).__name__} is not graphable.")

        if analysis.table is not self.table:
            raise ValueError("Linked analysis must be from the same DataTable")

        self._linked_analyses.append(analysis)
        return self

    @abstractmethod
    def draw(self, ax: Axes | None) -> Axes: ...

    @property
    def plot_properties(self) -> dict[str, PlotProps]:
        if not hasattr(self, "_plot_properties"):
            self._compile_plot_properties()
        return self._plot_properties

    def _compile_plot_properties(self) -> None:
        assert hasattr(self, "_table")

        schema_key = f"graphs.{self._schema_access_key()}"
        props = SG_DEFAULTS[schema_key]
        static_props = {k: v for k, v in props.items()
                        if not isinstance(v, list)}
        cycles = {k: cycle(v) for k, v in props.items() if isinstance(v, list)}
        dataset_props = {}

        for ds_id, *p in zip(self.table.dataset_ids, *cycles.values()):
            cyclic_props = dict(zip(cycles.keys(), p))
            dataset_props[ds_id] = PlotProps(**static_props, **cyclic_props)
        
        self._plot_properties = dataset_props

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

    def _register_component(
        self,
        ty: str,
        opt_t: Optional[type[Option]],
        component_t: type[GraphComponent],
        kw: dict[str, Any],
        **kwargs,
    ) -> None:
        if not opt_t:
            component = component_t.from_opt(None, kw, **kwargs)
        else:
            opt = opt_t.from_str(ty)
            component = component_t.from_opt(opt, kw, **kwargs)
        self._components.append(component)

    @classmethod
    def _schema_access_key(cls) -> str:
        return cls.__name__.lower().replace("graph", "")


class GraphComponent(ABC):

    def __init__(self, kw: dict[str, Any]) -> None:
        self.kw = kw

    @abstractmethod
    def draw_xy(self, graph: XYGraph, ax: Axes) -> None: ...

    @abstractmethod
    def draw_column(self, graph: ColumnGraph, ax: Axes) -> None: ...

    @classmethod
    @abstractmethod
    def from_opt(cls, opt, kw, **kwargs) -> Self: ...
