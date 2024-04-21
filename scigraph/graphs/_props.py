from dataclasses import dataclass


@dataclass
class PlotProps:
    linewidth: float
    markersize: float
    capsize: float
    capthickness: float
    color: str
    marker: str
    ls: str

    def point_kws(self) -> dict:
        return {
            "marker": self.marker,
            "color": self.color,
            "markersize": self.markersize,
            "ls": "",
        }

    def line_kws(self) -> dict:
        return {
            "color": self.color,
            "ls": self.ls,
            "linewidth": self.linewidth,
            "marker": "",
        }

    def errorbar_kw(self) -> dict:
        return {
            "ecolor": self.color,
            "elinewidth": self.linewidth,
            "capsize": self.capsize,
            "capthick": self.capthickness,
            "marker": "",
            "ls": ""
        }
