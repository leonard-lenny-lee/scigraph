from dataclasses import dataclass
from typing import Optional


@dataclass
class PlotProps:
    # Common properties
    linewidth: float
    markersize: float
    capsize: float
    capthickness: float
    markeredgecolor: str
    markerfacecolor: str
    marker: str
    ls: str

    # Bar properties
    barcolor: Optional[str] = None
    barwidth: Optional[float] = None
    baredgecolor: Optional[str] = None
    baredgethickness: Optional[float] = None

    def point_kws(self) -> dict:
        return {
            "marker": self.marker,
            "markeredgecolor": self.markeredgecolor,
            "markerfacecolor": self.markerfacecolor,
            "markersize": self.markersize,
            "ls": "",
            "markerfacecolor": "none",
        }

    def line_kws(self) -> dict:
        return {
            "color": self.markeredgecolor,
            "ls": self.ls,
            "linewidth": self.linewidth,
            "marker": "",
        }

    def errorbar_kw(self) -> dict:
        return {
            "ecolor": self.markeredgecolor,
            "elinewidth": self.linewidth,
            "capsize": self.capsize,
            "capthick": self.capthickness,
            "marker": "",
            "ls": "",
        }

    def bar_kw(self, vertical: bool = False) -> dict:
        out = {
            "color": self.barcolor,
            "linewidth": self.baredgethickness,
            "edgecolor": self.baredgecolor,
        }
        key = "width" if vertical else "height"
        out[key] = self.barwidth  # type: ignore
        return out

    def barl_kw(self) -> dict:
        return {
            "color": self.baredgecolor,
            "linewidth": self.baredgethickness,
        }

    def box_and_whisker_kw(self) -> dict:
        return {
            "widths": self.barwidth,
            "boxprops": {
                "color": self.markeredgecolor,
                "linewidth": self.baredgethickness,
                "linestyle": self.ls,
            },
            "whiskerprops": {
                "color": self.markeredgecolor,
                "linewidth": self.baredgethickness,
            },
            "flierprops": {
                "marker": self.marker,
                "color": self.markeredgecolor,
                "markersize": self.markersize,
            },
            "medianprops": {
                "color": self.baredgecolor,
                "linewidth": self.baredgethickness,
            },
        }
