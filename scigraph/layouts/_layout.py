from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING, Iterator, Literal
import logging

from matplotlib.figure import Figure
from matplotlib.legend import Legend

from scigraph.layouts._caption import Caption

if TYPE_CHECKING:
    from scigraph.graphs.abc import Graph


class Layout(ABC):

    def __init__(self) -> None:
        self._captions: list[Caption] = []
        self._create_layout_legend = False
        self._legend_artists: dict[str, list[Any]] = {}
        self._legend_kw = {}

    def link_graph(
        self,
        graph: Graph,
        key: Any | None,
    ) -> None:
        if key is None:
            key = self._get_empty_pos_key()
        if self.get_position(key) is not None:
            logging.warn(f"Replaced graph at position {key}.")
        self.set_position(key, graph)

    def add_caption(
        self,
        position: Literal["above", "right", "below", "left"]
    ) -> Caption:
        caption = Caption(position)
        self._captions.append(caption)
        return caption

    def add_layout_legend(self, **legend_kw) -> None:
        self._create_layout_legend = True
        self._legend_kw = legend_kw

    def draw(self, **fig_kw) -> Figure:
        fig = self._draw(**fig_kw)
        for caption in self._captions:
            caption.draw(fig)
        if self._create_layout_legend:
            self._draw_legend(fig)
        return fig

    @abstractmethod
    def _draw(self, **fig_kw) -> Figure: ...

    def _compose_legend_artists(self) -> None:
        """Aggregate all legend artists in linked graphs."""
        self._legend_artists = {}
        for graph in self.iter_positions():
            if graph is None:
                continue
            for label, artists in graph._legend_artists.items():
                if label in self._legend_artists:
                    self._legend_artists[label].extend(artists)
                else:
                    self._legend_artists[label] = artists

    def _draw_legend(self, fig: Figure) -> Legend:
        self._compose_legend_artists()
        labels = []
        handles = []
        for k, v in self._legend_artists.items():
            labels.append(k)
            handles.append(tuple(v))
        return fig.legend(handles, labels, **self._legend_kw)

    @abstractmethod
    def get_position(self, key: Any) -> Graph | None: ...

    @abstractmethod
    def set_position(self, key: Any, graph: Graph) -> None: ...

    @abstractmethod
    def iter_positions(self) -> Iterator[Graph | None]: ...

    @abstractmethod
    def iter_graphs(self) -> Iterator[Graph]: ...

    @abstractmethod
    def _get_empty_pos_key(self) -> Any: ...
