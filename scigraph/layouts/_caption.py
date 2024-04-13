from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Self, override

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

from scigraph.config import SG_DEFAULTS


class Caption:

    def __init__(
        self,
        position: Literal["above", "right", "below", "left"],
    ) -> None:
        self.position = position
        self._elements: list[CaptionElement] = []
        self.pad: dict[str, float] = SG_DEFAULTS["layout.caption.pad"]
        self.direction: Literal["down", "up"] = SG_DEFAULTS["layout.caption.direction"]

    def add_text(
        self,
        s: str,
        ty: Literal["h1", "h2", "p", "s"],
        *,
        size: float | None = None,
        weight: str | None = None,
        color: str | None = None,
        spacing_before: float | None = None,
        spacing_after: float | None = None,
        linespacing: float | None = None,
        vertical_alignment : str | None = None,
        horizontal_alignment : str | None = None,
    ) -> Self:
        key = "layout.caption.text."
        if ty not in ["h1", "h2", "p", "s"]:
            raise ValueError(f"Invalid type {ty}")
        key += ty
        cfg = SG_DEFAULTS[key]
        assert isinstance(cfg, dict)
        kwargs = locals()
        for k in cfg:
            if kwargs[k] is not None:
                cfg[k] = kwargs[k]
        self._elements.append(TextElement(s, **cfg))
        return self

    def add_line(
        self,
        ty: Literal["dividing", "branding"],
        *,
        length: float | None = None,
        width: float | None = None,
        color: str | None = None,
        spacing_before: float | None = None,
        spacing_after: float | None = None,
    ) -> Self:
        if ty not in ["dividing", "branding"]:
            raise ValueError(f"Invalid type {ty}")
        key = f"layout.caption.line.{ty}"
        cfg = SG_DEFAULTS[key]
        kwargs = locals()
        for k in cfg:
            if kwargs[k] is not None:
                cfg[k] = kwargs[k]
        self._elements.append(LineElement(**cfg))
        return self

    def draw(self, fig: Figure) -> None:
        # First draw all in the top left
        width, height = fig.get_size_inches()
        x, y = 0, 1
        x += pt_to_coordinate_delta(self.pad["left"], width)
        y -= pt_to_coordinate_delta(self.pad["top"], height)
        max_width = 0
        artists: list[MPLArtistWrapper] = []

        for e in self._elements:
            spacing_before_pt, spacing_after_pt = e.vpad_pt
            y -= pt_to_coordinate_delta(spacing_before_pt, height)
            artist = e.draw(fig, x, y)
            artist_width, artist_height = artist.size()
            max_width = max(max_width, artist_width)
            y -= artist_height
            y -= pt_to_coordinate_delta(spacing_after_pt, height)
            artists.append(artist)
        y -= pt_to_coordinate_delta(self.pad["bottom"], height)

        # Translate according to caption position
        total_x_delta = x + max_width \
            + pt_to_coordinate_delta(self.pad["right"], width)
        total_y_delta = 1 - y
        translate_x, translate_y = 0, 0

        match self.position:
            case "above":
                translate_y = total_y_delta
            case "below":
                translate_y = -1
            case "right":
                translate_x = 1
            case "left":
                translate_x = -total_x_delta
            case _:
                raise ValueError

        for a in artists:
            a.translate(translate_x, translate_y)


class CaptionElement(ABC):

    @abstractmethod
    def draw(self, fig: Figure, x: float, y: float) -> MPLArtistWrapper: ...

    @property
    @abstractmethod
    def vpad_pt(self) -> tuple[float, float]: ...


@dataclass
class TextElement(CaptionElement):
    s: str
    size: float
    weight: str
    color: str
    linespacing: float
    spacing_before: float
    spacing_after: float
    vertical_alignment : str
    horizontal_alignment : str

    @override
    def draw(self, fig: Figure, x: float, y: float) -> MPLArtistWrapper:
        text = fig.text(
            x, y, self.s,
            size=self.size,
            weight=self.weight,
            color=self.color,
            linespacing=self.linespacing,
            va=self.vertical_alignment,
            ha=self.horizontal_alignment
        )
        return TextWrapper(text)

    @property
    @override
    def vpad_pt(self) -> tuple[float, float]:
        return self.spacing_before, self.spacing_after


@dataclass
class LineElement(CaptionElement):
    length: float
    width: float
    color: str
    spacing_before: float
    spacing_after: float

    @override
    def draw(self, fig: Figure, x: float, y: float) -> MPLArtistWrapper:
        width, _ = fig.get_size_inches()
        x_delta = pt_to_coordinate_delta(self.length, width)
        line = Line2D(
            [x, x + x_delta],
            [y, y],
            color=self.color,
            linewidth=self.width,
            linestyle="-",
            marker="",
        )
        fig.add_artist(line)
        return Line2DWrapper(line)

    @property
    @override
    def vpad_pt(self) -> tuple[float, float]:
        return self.spacing_before, self.spacing_after


class MPLArtistWrapper[T](ABC):  # Adaptor to provide uniform API

    def __init__(self, artist: T) -> None:
        self.artist = artist

    @abstractmethod
    def translate(self, x: float, y: float) -> None: ...

    @abstractmethod
    def size(self) -> tuple[float, float]: ...


class Line2DWrapper(MPLArtistWrapper[Line2D]):

    @override
    def translate(self, x: float, y: float) -> None:
        cur_xdata, cur_ydata = self.artist.get_data()
        new_xdata = np.array(cur_xdata) + x
        new_ydata = np.array(cur_ydata) + y
        self.artist.set_data(new_xdata, new_ydata)

    @override
    def size(self) -> tuple[float, float]:
        bbox = self.artist.get_bbox()
        return bbox.width, bbox.height
        

class TextWrapper(MPLArtistWrapper[Text]):

    @override
    def translate(self, x: float, y: float) -> None:
        cur_x, cur_y = self.artist.get_position()
        self.artist.set_position((cur_x + x, cur_y + y))

    @override
    def size(self) -> tuple[float, float]:
        bbox = self.artist.get_tightbbox()
        fig = self.artist.get_figure()
        assert bbox is not None and fig is not None
        fig_w, fig_h = fig.get_size_inches()
        width = pxl_to_coordinate_delta(bbox.width, fig_w, fig.dpi)
        height = pxl_to_coordinate_delta(bbox.height, fig_h, fig.dpi)
        return width, height


def pt_to_coordinate_delta(
    pt: float,
    fig_length_inches: float
) -> float:
    MPL_PT_PER_INCH = 72
    inches = pt / MPL_PT_PER_INCH
    return inches / fig_length_inches


def pxl_to_coordinate_delta(
    pxl: float,
    fig_length_inches: float,
    fig_dpi: float,
) -> float:
    inches = pxl / fig_dpi
    return inches / fig_length_inches
