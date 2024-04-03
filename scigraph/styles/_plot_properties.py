from dataclasses import dataclass
from itertools import cycle

import matplotlib.pyplot as plt

import scigraph.styles.defaults as defaults


@dataclass
class PlotProperties:
    color: str
    marker: str
    linestyle: str


def generate_plot_prop_cycle(length: int) -> list[PlotProperties]:
    mpl_prop_cycle = plt.rcParams["axes.prop_cycle"]
    out = []
    for _, props in zip(range(length), cycle(mpl_prop_cycle)):
        color = props.get("color", defaults.COLOR)
        marker = props.get("marker", defaults.MARKER)
        ls = props.get("ls", defaults.LINESTYLE)
        out.append(PlotProperties(color=color, marker=marker, linestyle=ls))
    return out
