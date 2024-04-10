from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, Iterator, override
import logging

from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from scigraph.graphs.abc import Graph


class Layout(ABC):

    # Default config - Currently quite hacky
    # TODO - Refactor into defaults module, configurable with TOML
    title_size_pt = 10
    title_weight = "bold"
    subtitle_size_pt = 8
    subtitle_weight = "bold"
    header_text_size_pt = 8
    header_weight = "regular"
    footer_text_size_pt = 6
    footer_weight = "regular"
    header_pad_pt = 2
    footer_pad_pt = 2
    left_pad_pt = 12
    spacing_before_pt = 2
    spacing_after_pt = 2
    linespacing_multiple = 1.1
    horizontal_alignment = "left"
    vertical_alignment = "center"
    line_color = "#002147"
    dividing_line_length_pt = 50
    dividing_line_width_pt = 0.5
    dividing_line_spacing_before_pt = 6
    dividing_line_spacing_after_pt = 6
    branding_line_length_pt = 20
    branding_line_width_pt = 4
    legend_ncols = -1

    def __init__(self) -> None:
        self._header: list[LayoutArtist] = []
        self._footer: list[LayoutArtist] = []
        self.create_header_legend = False
        self._legend_artists: dict[str, list[Any]] = {}

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

    def add_title(self, s: str, **kwargs) -> None:
        self.add_header_text(s, weight=self.title_weight,
                             size_pt=self.title_size_pt, **kwargs)

    def add_subtitle(self, s: str, **kwargs) -> None:
        self.add_header_text(s, weight=self.subtitle_weight,
                             size_pt=self.subtitle_size_pt, **kwargs)

    def add_header_text(self, s: str, **kwargs) -> None:
        self._header.append(HeaderTextArtist(s, **kwargs))

    def add_footer_text(self, s: str, **kwargs) -> None:
        self._footer.append(FooterTextArtist(s, **kwargs))

    def add_dividing_line(
        self,
        length: float | None = None,
        width: float | None = None,
        **kwargs
    ) -> None:
        if length is None:
            length = self.dividing_line_length_pt
        if width is None:
            width = self.dividing_line_width_pt
        self._header.append(
            HorizontalLineArtist(
                self.line_color,
                length,
                width,
                self.dividing_line_spacing_before_pt,
                self.dividing_line_spacing_after_pt,
                kwargs
            )
        )

    def add_branding_line(
        self,
        length: float | None = None,
        width: float | None = None,
        **kwargs
    ) -> None:
        if length is None:
            length = self.branding_line_length_pt
        if width is None:
            width = self.branding_line_width_pt
        self._header.append(
            HorizontalLineArtist(
                self.line_color,
                length,
                width,
                self.spacing_before_pt,
                self.spacing_after_pt,
                kwargs
            )
        )

    @abstractmethod
    def draw(self, **fig_kw) -> tuple[Figure, Any]: ...

    def _draw_header(self, fig: Figure) -> None:
        fig_width, fig_height = fig.get_size_inches()
        x = pt_to_fig_coordinate_delta(self.left_pad_pt, fig_width)
        # Draw from bottom to top
        cur_y = 1
        pad_delta = pt_to_fig_coordinate_delta(self.header_pad_pt, fig_height)
        cur_y += pad_delta
        # Always plot legend first
        if self.create_header_legend:
            self._compose_legend_artists()
            legend = self._draw_header_legend(fig, cur_y)
            fig_height_pixels = fig.dpi * fig_height
            cur_y = legend.get_window_extent().ymax / fig_height_pixels
            cur_y += pad_delta
        for artist in reversed(self._header):
            height = artist.calculate_height(fig_height)
            y = cur_y + height * artist.horizontal_centering_ratio()
            artist.draw(fig, x, y)
            cur_y += height

    def _draw_footer(self, fig: Figure) -> None:
        fig_width, fig_height = fig.get_size_inches()
        x = pt_to_fig_coordinate_delta(
            self.left_pad_pt, fig_width,
        )
        # Draw from top to bottom
        cur_y = 0
        cur_y -= pt_to_fig_coordinate_delta(self.footer_pad_pt, fig_height)
        for artist in self._footer:
            height = artist.calculate_height(fig_height)
            cur_y -= height
            y = cur_y + height * artist.horizontal_centering_ratio()
            artist.draw(fig, x, y)

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

    def _draw_header_legend(self, fig: Figure, y: float) -> Legend:
        labels = []
        handles = []
        for k, v in self._legend_artists.items():
            labels.append(k)
            handles.append(tuple(v))
        
        ncols = self.legend_ncols if self.legend_ncols > 0 else len(labels)
        return fig.legend(
            handles,
            labels,
            ncols=ncols,
            loc="center",
            bbox_to_anchor=(0.5, y),
        )

    @abstractmethod
    def get_position(self, key: Any) -> Graph | None: ...

    @abstractmethod
    def set_position(self, key: Any, graph: Graph) -> None: ...

    @abstractmethod
    def iter_positions(self) -> Iterator[Graph | None]: ...

    @abstractmethod
    def _get_empty_pos_key(self) -> Any: ...


class LayoutArtist(ABC):

    @abstractmethod
    def calculate_height(self, fig_height_inches: float) -> float:
        pass

    @abstractmethod
    def horizontal_centering_ratio(self) -> float: ...

    @abstractmethod
    def draw(self, fig: Figure, x: float, y: float) -> None:
        pass


@dataclass
class TextArtist(LayoutArtist):

    s: str
    weight: str 
    size_pt: float
    spacing_before_pt: float
    spacing_after_pt: float
    linespacing: float
    ha: str
    va: str
    txt_kwargs: dict[str, Any]

    @override
    def calculate_height(self, fig_height_inches: float) -> float:
        height_pt = (
            self.spacing_before_pt
            + self.spacing_after_pt
            + self.size_pt
        ) * self.linespacing
        return pt_to_fig_coordinate_delta(height_pt, fig_height_inches)

    @override
    def horizontal_centering_ratio(self) -> float:
        # Calculate from bottom
        total_height = self.size_pt + self.spacing_before_pt + self.spacing_after_pt
        text_midpoint = self.spacing_after_pt + self.size_pt / 2
        return text_midpoint / total_height

    @override
    def draw(self, fig: Figure, x: float, y: float) -> None:
        fig.text(
            x,
            y,
            self.s,
            size=self.size_pt,
            weight=self.weight,
            linespacing=self.linespacing,
            ha=self.ha,
            va=self.va,
            **self.txt_kwargs,
        )


@dataclass
class HeaderTextArtist(TextArtist):

    s: str
    weight: str = Layout.header_weight
    size_pt: float = Layout.header_text_size_pt
    spacing_before_pt: float = Layout.spacing_before_pt
    spacing_after_pt: float = Layout.spacing_after_pt
    linespacing: float = Layout.linespacing_multiple
    ha: str = Layout.horizontal_alignment
    va: str = Layout.vertical_alignment
    txt_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class FooterTextArtist(TextArtist):

    s: str
    weight: str = Layout.footer_weight
    size_pt: float = Layout.footer_text_size_pt
    spacing_before_pt: float = Layout.spacing_before_pt
    spacing_after_pt: float = Layout.spacing_after_pt
    linespacing: float = Layout.linespacing_multiple
    ha: str = Layout.horizontal_alignment
    va: str = Layout.vertical_alignment
    txt_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class HorizontalLineArtist(LayoutArtist):

    color: str
    length_pt: float
    width_pt: float
    spacing_before_pt: float
    spacing_after_pt: float
    line_kwargs: dict

    @override
    def calculate_height(self, fig_height_inches: float) -> float:
        height = self.width_pt + self.spacing_before_pt + self.spacing_after_pt
        return pt_to_fig_coordinate_delta(height, fig_height_inches)

    @override
    def horizontal_centering_ratio(self) -> float:
        # From bottom
        height = self.width_pt + self.spacing_before_pt + self.spacing_after_pt
        line_midpoint = self.spacing_after_pt + self.width_pt / 2
        return line_midpoint / height

    @override
    def draw(self, fig: Figure, x: float, y: float) -> None:
        width, _ = fig.get_size_inches()
        fig.add_artist(
            Line2D(
                (x, x + pt_to_fig_coordinate_delta(self.length_pt, width)),
                (y, y),
                linewidth=self.width_pt,
                color=self.color,
                **self.line_kwargs,
            )
        )


def pt_to_fig_coordinate_delta(
    pt: float,
    total_length_inches: float
) -> float:
    MPL_PT_PER_INCH = 72
    inches = pt / MPL_PT_PER_INCH
    return inches / total_length_inches
