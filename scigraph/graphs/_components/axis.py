from matplotlib.axis import Axis as MPLAxis

from .._formatters import LOG10_TICK_FORMATTERS, LINEAR_TICK_FORMATTERS


class Axis:

    valid_scale_opts = {"linear", "log10"}
    valid_format_opts = {
        "linear": {"decimal", "power10", "antilog"},
        "log10": {"log10", "power10", "antilog"},
    }
    default_format_opt = {
        "linear": "decimal",
        "log10": "power10"
    }
    tick_fmt_fns = {
        "linear": LINEAR_TICK_FORMATTERS,
        "log10": LOG10_TICK_FORMATTERS,
    }
    mpl_scale_map = {
        "linear": "linear",
        "log10": "log",
    }

    def __init__(
        self,
        title: str = None,
        scale: str = "linear",
        format: str = None
    ) -> None:
        self.title = title
        self._scale = None
        self._format = None
        self.scale = scale
        if format is None:
            format = self._default_format()
        self.format = format

    def format_axis(self, axis: MPLAxis) -> None:
        axis._set_axes_scale(self._mpl_scale)
        fmt_fn = self.tick_fmt_fns[self.scale][self.format]
        fmt_fn(axis)

    def _default_format(self) -> str:
        if self.scale not in self.default_format_opt:
            raise NotImplementedError
        return self.default_format_opt[self.scale]

    @property
    def scale(self) -> str:
        return self._scale

    @scale.setter
    def scale(self, val: str) -> None:
        if val in self.valid_scale_opts:
            self._scale = val
        else:
            raise ValueError(
                f"Invalid scale argument. Options: {self.valid_scale_opts}"
            )
        if self.format not in self.valid_format_opts[val]:
            self.format = self._default_format()

    @property
    def _mpl_scale(self) -> str:
        return self.mpl_scale_map[self.scale]

    @property
    def format(self) -> str:
        return self._format

    @format.setter
    def format(self, val: str) -> None:
        if self.scale not in self.valid_format_opts:
            raise NotImplementedError
        valid_format_opts = self.valid_format_opts[self.scale]
        if val in valid_format_opts:
            self._format = val
        else:
            raise ValueError(
                f"Invalid scale argument. Options: {valid_format_opts}"
            )
