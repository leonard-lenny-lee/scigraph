import math
from typing import Callable

from matplotlib.axis import Axis
from matplotlib.ticker import FuncFormatter


def set_decimal_format(_: Axis) -> None:
    # Dummy
    pass


def set_antilog_format(axis: Axis) -> None:
    formatter = FuncFormatter(lambda x, _:
                              "%s" % int(x) if math.log10(x) > 0 else x)
    axis.set_major_formatter(formatter)


def set_power10_format(axis: Axis) -> None:
    formatter = FuncFormatter(lambda x, _: "$10^{%s}$" % int(math.log10(x)))
    axis.set_major_formatter(formatter)


def set_log10_format(axis: Axis) -> None:
    formatter = FuncFormatter(lambda x, _: "%s" % int(math.log10(x)))
    axis.set_major_formatter(formatter)


LOG10_TICK_FORMATTERS: dict[str, Callable[[Axis], None]] = {
    "antilog": set_antilog_format,
    "power10": set_power10_format,
    "log10": set_log10_format,
}

LINEAR_TICK_FORMATTERS: dict[str, Callable[[Axis], None]] = {
    "decimal": set_decimal_format,
    "antilog": set_antilog_format,
    "power10": set_power10_format,
}
