from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from . import cfg


class Graph(ABC):

    @abstractmethod
    def __init__(self) -> None:
        self._cfg = None
        self._fig = None
        self._axes = None
        self.x_label = None
        self.y_label = None

    @abstractmethod
    def plot(self, *args, **kwargs) -> None: ...

    def show(self):
        if self._fig is None:
            raise NameError("fig not initialized, call plot()")
        # self._fig.show()
        self.fig.show()

    @abstractmethod
    def _plot_axes(self, ax: Axes) -> None: ...

    def add_config(self, key: str, val: Any):
        if key not in self.cfg.keys:
            raise ValueError(f"Unrecognised config key: '{key}'")
        if self._cfg is None:
            # Create a local copy of the default config object
            self._cfg = deepcopy(cfg)
        self._cfg[key] = val
        return self

    @property
    def cfg(self):
        if self._cfg is None:
            return cfg
        else:
            return self._cfg

    @property
    def fig(self) -> Figure:
        if self._fig is None:
            raise NameError("fig not initialized, call plot()")
        return self._fig

    @property
    def axes(self) -> Axes:
        if self._axes is None:
            raise NameError("axes not initialized, call plot()")
        return self._axes

    def _apply_axes_style(self, ax: Axes) -> None:
        # Format spine visibility
        for k, v in self.cfg["axes.spines.visible"].items():
            ax.spines[k].set_visible(v)
        # Add axis labels
        if self.cfg["axes.label_x"]:
            ax.set_xlabel(self.x_label)
        if self.cfg["axes.label_y"]:
            ax.set_ylabel(self.y_label)
