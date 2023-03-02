from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from matplotlib.axes import Axes
from . import cfg


class Graph(ABC):

    @abstractmethod
    def __init__(self) -> None:
        self._cfg = None

    @abstractmethod
    def plot(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def _plot_axes(self, ax: Axes) -> None:
        pass

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

    def _apply_axes_style(self, ax: Axes) -> None:
        # Format spine visibility
        for k, v in self.cfg["axes.spines.visible"].items():
            ax.spines[k].set_visible(v)
