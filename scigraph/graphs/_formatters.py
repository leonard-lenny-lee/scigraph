import math

from matplotlib.axis import Axis
from matplotlib.ticker import FuncFormatter


def set_antilog_format(axis: Axis) -> None:
    formatter = FuncFormatter(lambda x, pos:
                              "%s" % int(x) if math.log10(x) > 0 else x)
    axis.set_major_formatter(formatter)


def set_power10_format(axis: Axis) -> None:
    formatter = FuncFormatter(lambda x, pos: "$10^{%s}$" % int(math.log10(x)))
    axis.set_major_formatter(formatter)


def set_log10_format(axis: Axis) -> None:
    formatter = FuncFormatter(lambda x, pos: "%s" % int(math.log10(x)))
    axis.set_major_formatter(formatter)
