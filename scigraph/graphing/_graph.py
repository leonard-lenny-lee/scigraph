"""Contains the abstract Graph class
"""
from __future__ import annotations

__all__ = ["Graph"]

from abc import ABC, abstractmethod
from enum import Enum

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.lines as mlines

from ..tables import DataTable
from .._utils.descriptors import *


class Graph(ABC):

    figsize = CfgProperty("figsize", "graph.figure.figsize")
    dpi = CfgProperty("dpi", "graph.figure.dpi")
    layout = CfgProperty("layout", "graph.figure.layout")
    spines = CfgProperty("spines", "graph.spines")
    cglpalette = SnsPalette("cglpalette", "graph.palettes.categorical")

    @abstractmethod
    def __init__(self) -> None:
        # Init attributes shared by all graphs
        self._fig = None
        self._axes = None
        self._group_names = []
        self.xlabel: str = None
        self.ylabel: str = None
        self.show_xlabel: bool = True
        self.show_ylabel: bool = True
        self.legend: bool = True

    @abstractmethod
    def plot(self, *args, **kwargs) -> None: ...

    def show(self):
        if self._fig is None:
            raise NameError("fig not initialized, call plot()")
        self.fig.show()

    @abstractmethod
    def _plot_axes(self, ax: Axes) -> None: ...

    @property
    @abstractmethod
    def dt(self) -> DataTable: ...

    @dt.setter
    @abstractmethod
    def dt(self, dt: DataTable) -> None: ...

    @property
    def fig(self) -> Figure:
        if self._fig is None:
            raise AttributeError("fig not initialized, call plot()")
        return self._fig

    @property
    def axes(self) -> Axes:
        if self._axes is None:
            raise AttributeError("axes not initialized, call plot()")
        return self._axes

    @property
    def fig_kw(self):
        return {
            "figsize": self.figsize,
            "dpi": self.dpi,
            "layout": self.layout,
        }

    def _set_kwargs(self, **kwargs) -> None:
        # Allow provided kwargs to override default attributes
        # Call at the end of init sequence
        for kw, arg in kwargs.items():
            if not hasattr(self, kw):
                raise ValueError(f"Invalid kw argument: {kw}")
            setattr(self, kw, arg)

    def _apply_axes_style(self, ax: Axes) -> None:
        # Add axis labels
        if self.show_xlabel:
            ax.set_xlabel(self.xlabel)
        if self.show_ylabel:
            ax.set_ylabel(self.ylabel)
        # Style spines
        [ax.spines[k].set_visible(v) for k, v in self.spines.items()]
        if self.legend:
            # # Generate legend styling
            handles = []
            for label, color in zip(self._group_names, self.cglpalette):
                handles.append(mlines.Line2D(
                    [], [], color=color, linestyle=self.linestyle,
                    marker=self.marker, label=label))
            ax.legend(handles=handles)
