from abc import ABC, abstractmethod

from matplotlib.axes import Axes
from . import cfg


class Graph(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def plot(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def _plot_axes(self, ax: Axes) -> None:
        pass

    def _apply_axes_def_style(self, ax: Axes) -> None:
        # Format spine visibility
        for k, v in cfg["axes.spines.visible"].items():
            ax.spines[k].set_visible(v)
