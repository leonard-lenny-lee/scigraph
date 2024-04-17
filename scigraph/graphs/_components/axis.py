from abc import ABC, abstractmethod
from typing import Literal, Optional, override

from matplotlib.axes import Axes

from .._formatters import TICK_FORMATTERS
from scigraph.config import SG_DEFAULTS
from scigraph._options import Option


class SGAxis(ABC):

    class _Axis(Option):
        X = 0
        Y = 1

    @abstractmethod
    def _format_axes(self, ax: Axes) -> None: ...


class ContinuousAxis(SGAxis):

    def __init__(
        self,
        axis: Literal["x", "y"],
        title: str = "",
        scale: Optional[str] = None,
        format: Optional[str] = None,
    ) -> None:
        self.scale = scale
        self.format = format
        self.title = title
        self._axis = self._Axis.from_str(axis)

    @override
    def _format_axes(self, ax: Axes) -> None:
        if self._axis is self._Axis.X:
            ax.set_xscale(self.scale_mpl_arg)
            ax.set_xlabel(self.title)
            axis = ax.xaxis
        else:
            ax.set_yscale(self.scale_mpl_arg)
            ax.set_ylabel(self.title)
            axis = ax.yaxis
        TICK_FORMATTERS[self.format](axis)

    @property
    def scale(self) -> str:
        return self._scale

    @scale.setter
    def scale(self, val: str | None) -> None:
        key = "graphs.axis.continuous.scale"
        if val is None:
            val = SG_DEFAULTS[key]
        else:
            scale_param = SG_DEFAULTS.query_schema(key)
            if val not in scale_param.opt:
                raise ValueError(
                    f"Invalid scale argument. Options: {scale_param.opt}"
                )
        assert isinstance(val, str)
        self._scale = val
        
        # Change which format if current is not longer valid
        if (hasattr(self, "_format") 
                and self._format not in self.allowed_formats):
            self._format = self._default_fmt

    @property
    def scale_mpl_arg(self) -> str:
        if self.scale == "log10":
            return "log"
        else:
            return self.scale

    @property
    def _default_fmt(self) -> str:
        return SG_DEFAULTS[f"graphs.axis.continuous.format.{self._scale}"]

    @property
    def allowed_formats(self) -> set[str]:
        return SG_DEFAULTS.query_schema(
            f"graphs.axis.continuous.format.{self._scale}"
        ).opt

    @property
    def format(self) -> str:
        return self._format

    @format.setter
    def format(self, val: str | None) -> None:
        if val is None:
            val = self._default_fmt
        else:
            if val not in self.allowed_formats:
                raise ValueError(
                    f"Invalid format argument. Options: {self.allowed_formats}"
                )
        self._format = val


class CategoricalAxis(SGAxis):

    def __init__(
        self,
        axis: Literal["x", "y"],
        categories: list[str],
        title: str = "",
    ) -> None:
        self._axis = self._Axis.from_str(axis)
        self._categories = categories
        self.title = title

    @override
    def _format_axes(self, ax: Axes) -> None:
        n_categories = len(self._categories)
        ticks = list(range(n_categories))
        lims = ticks[0] - 0.49, ticks[-1] + 0.49

        if self._axis is self._Axis.X:
            axis = ax.xaxis
            ax.set_xlim(*lims)
        else:  # Y Axis
            axis = ax.yaxis
            ax.set_ylim(*lims)

        axis.set_ticks(ticks, self._categories)

