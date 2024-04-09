from typing import NamedTuple

from matplotlib.axis import Axis as MPLAxis

from .._formatters import (
    LOG10_TICK_FORMATTERS,
    LINEAR_TICK_FORMATTERS
)


class _ScaleProperties(NamedTuple):
    valid_format_opts: set[str]
    default_format_opt: str
    tick_fmt_fns: dict
    mpl_arg: str


class Axis:

    valid_scale_opts = {"linear", "log10"}
    scale_properties = {
        "linear": _ScaleProperties(
            valid_format_opts={"decimal", "power10", "antilog"},
            default_format_opt="decimal",
            tick_fmt_fns=LINEAR_TICK_FORMATTERS,
            mpl_arg="linear",
        ),
        "log10": _ScaleProperties(
            valid_format_opts={"log10", "power10", "antilog"},
            default_format_opt="power10",
            tick_fmt_fns=LOG10_TICK_FORMATTERS,
            mpl_arg="log",
        )
    }

    def __init__(
        self,
        title: str = "",
        scale: str = "linear",
        format: str | None = None
    ) -> None:
        self._format = None
        self.title = title
        self.scale = scale
        if format is None:
            format = self._props.default_format_opt
        self.format = format

    def format_axis(self, axis: MPLAxis) -> None:
        fmt_fn = self._props.tick_fmt_fns[self.format]
        fmt_fn(axis)

    @property
    def _props(self) -> _ScaleProperties:
        if self.scale not in self.scale_properties:
            raise NotImplementedError
        return self.scale_properties[self.scale]

    @property
    def scale(self) -> str:
        return self._scale

    @scale.setter
    def scale(self, val: str) -> None:
        if val not in self.valid_scale_opts:
            raise ValueError(
                f"Invalid scale argument. Options: {self.valid_scale_opts}"
            )
        self._scale = val
        if self.format is None \
                or self.format not in self._props.valid_format_opts:
            self.format = self._props.default_format_opt

    @property
    def format(self) -> str:
        if self._format is None:
            return self._props.default_format_opt
        return self._format

    @format.setter
    def format(self, val: str) -> None:
        if val not in self._props.valid_format_opts:
            raise ValueError(
                "Invalid scale argument. Options: "
                + ", ".join(self._props.valid_format_opts)
            )
        self._format = val
