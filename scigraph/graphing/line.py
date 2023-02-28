"""
"""
from typing import Tuple
import matplotlib.pyplot as plt
from numpy import ndarray, tile
from .graph import Graph
from ..tables import XYTable
from ..utils.args import Tokenizer

Z_VALUE_95 = 1.960


class LineGraph(Graph):

    def __init__(
        self,
        dt: XYTable,
        avg: str = "mean",
        err: str = "std",
    ):
        self.dt = dt
        self.avg = _Avg.from_str(avg)
        self.err = _Err.from_str(err)
        self._check_dtype()
        self._init_points()
        self._init_err()

    def plot(
        self,
        figsize: Tuple[float, float] = (8.0, 6.0),
        dpi: float = 100.0,
    ) -> None:
        fig_kw = {
            "figsize": figsize,
            "dpi": dpi,
        }
        if self.err is _Err.ALL or self.err is _Err.NONE:
            fig, ax = self._all_none_plot(fig_kw)
        else:
            fig, ax = self._err_plot(fig_kw)
        ax.set_xlabel(self.dt.x_name)
        fig.show()

    @property
    def _grouped(self):
        return self.dt.data.groupby(axis=1, level=0, sort=False)

    @property
    def _std(self) -> ndarray:
        return self._grouped.std(numeric_only=True).values

    @property
    def _count(self) -> ndarray:
        return self._grouped.count().values

    @property
    def _sem(self) -> ndarray:
        return self._std / (self._count ** 0.5)

    def _check_dtype(self) -> None:
        if not isinstance(self.dt, XYTable):
            raise ValueError(f"Only XYTables can be used with LineGraph")

    def _init_points(self) -> None:
        """Calculate XY data points"""
        if self.avg is _Avg.MEAN:
            self._points = self._grouped.mean().values
        elif self.avg is _Avg.MEDIAN:
            self._points = self._grouped.median().values

    def _init_err(self) -> None:
        """Calculate error bars for datapoints"""
        if self.err is _Err.STD:
            self._n_err = self._p_err = self._std
        elif self.err is _Err.SEM:
            self._n_err = self._p_err = self._sem
        elif self.err is _Err.CI95:
            self._p_err = self._n_err = Z_VALUE_95 * self._sem
        elif self.err is _Err.RANGE:
            self._p_err = self._grouped.max().values - self._points
            self._n_err = self._points - self._grouped.min().values
        else:  # NONE or ALL
            self._p_err = self._n_err = None

    def _err_plot(self, fig_kw):
        """Generate plot with error bars"""
        assert self._p_err is not None
        fig, ax = plt.subplots(**fig_kw)
        # Transpose to allow convenient indexing
        points, n_err, p_err = self._points.T, self._n_err.T, self._p_err.T
        x, *x_err = points[0], n_err[0], p_err[0]
        y, y_n_err, y_p_err = points[1:], n_err[1:], p_err[1:]
        plt_data = (self.dt.group_names, y, y_n_err, y_p_err)
        # Assert the correct number of groups has been preserved
        assert len({len(f) for f in plt_data}) == 1
        for group_name, y, *y_err in zip(*plt_data):
            ax.plot(x, y, label=group_name)
            ax.errorbar(x, y, y_err, x_err, fmt="none")
        return fig, ax

    def _all_none_plot(self, fig_kw):
        """Generate plot with no error bars or individual points plotted"""
        assert self.err is _Err.ALL or self.err is _Err.NONE
        fig, ax = plt.subplots(**fig_kw)
        points = self._points.T
        x, y = points[0], points[1:]
        y_all = self.dt.data.values.T[self.dt.n_x_replicates:]
        for i, (group_name, y) in enumerate(zip(self.dt.group_names, y)):
            # Plot averages at each X as line
            ax.plot(x, y, label=group_name)
            if self.err is _Err.NONE:
                continue
            # Plot individual data points as scatter if 'all' is specified
            n = self.dt.n_y_replicates
            ax.scatter(x=tile(x, n), y=y_all[i*n:i*n+n].ravel())
        return fig, ax


class _Avg(Tokenizer):

    MEAN = 0
    MEDIAN = 1


class _Err(Tokenizer):

    STD = 0
    SEM = 1
    CI95 = 2
    RANGE = 3
    ALL = 4
    NONE = 5
