"""Contains the abstract Graph class
"""
from __future__ import annotations

__all__ = ["Graph"]

from abc import ABC, abstractmethod

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..tables import DataTable


class Graph(ABC):

    @abstractmethod
    def __init__(self) -> None:
        # Init attributes shared by all graphs
        self._fig = None
        self._axes = None
        self.x_label: str = None
        self.y_label: str = None
        self.label_x: bool = True
        self.label_y: bool = True
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

    def _set_kwargs(self, **kwargs) -> None:
        # Allow provided kwargs to override default attributes
        # Call at the end of init sequence
        for kw, arg in kwargs.items():
            if not hasattr(self, kw):
                raise ValueError(f"Invalid kw argument: {kw}")
            setattr(self, kw, arg)

    def _apply_axes_style(self, ax: Axes) -> None:
        # Add axis labels
        if self.label_x:
            ax.set_xlabel(self.x_label)
        if self.label_y:
            ax.set_ylabel(self.y_label)
        if self.legend:
            ax.legend()
