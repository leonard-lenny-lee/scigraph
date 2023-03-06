"""Contains the LineGraph class.
"""

from __future__ import annotations
from typing import Tuple
from warnings import warn

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from numpy import ndarray, tile

from . import cfg
from .graph import Graph
from ..tables import XYTable
from ..utils.args import Token


class LineGraph(Graph):

    def __init__(
        self,
        dt: XYTable,
        avg: str = "mean",
        err: str = "std",
        ci: float = 0.95,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.avg = avg
        self.err = err
        self.ci = ci
        self.x_label = self.dt.x_label
        self.y_label = self.dt.y_label

    def plot(self) -> None:
        self._fig, self._axes = plt.subplots(**self.cfg["figure"])
        self._plot_axes(self._axes)

    def _plot_axes(self, ax: Axes) -> None:
        if self.err is _Err.ALL or self.err is _Err.NONE:
            self._all_none_plot(ax)
        else:
            self._err_plot(ax)
        self._apply_axes_style(ax)

    @property
    def dt(self) -> XYTable:
        return self._dt

    @dt.setter
    def dt(self, dt: XYTable) -> None:
        if not isinstance(dt, XYTable):
            raise ValueError(f"Only XYTables can be used with LineGraph")
        self._dt = dt

    @property
    def avg(self) -> _Avg:
        return self._avg

    @avg.setter
    def avg(self, avg: str) -> None:
        self._avg = _Avg.from_str(avg)

    @property
    def err(self) -> _Err:
        return self._err

    @err.setter
    def err(self, err: str) -> None:
        """Emit warnings if specified errors are incompatible with data"""
        err = _Err.from_str(err)
        if self.dt.n_y_replicates == 1 and self.dt.n_x_replicates == 1:
            warn(
                f"DataTable contains no replicates to plot error with "
                f"method '{err}'. Defaulting to 'none'. ",
                RuntimeWarning
            )
            err = _Err.NONE
        if self.dt.n_x_replicates > 1 and err is _Err.ALL:
            warn(
                "Scigraph unable to plot more than one x replicate. "
                "Plotting x values as an average of all x replicates. ",
                RuntimeWarning
            )
        self._err = err

    @property
    def ci(self) -> float:
        return self._ci

    @ci.setter
    def ci(self, value: float) -> None:
        if 0 < value < 1:
            self._ci = value
            return
        raise ValueError("ci must take a value between 0 and 1")

    def _calc_avg(self) -> ndarray:
        """Calculate XY data points"""
        if self.avg is _Avg.MEAN:
            points = self.dt.mean.values
        elif self.avg is _Avg.MEDIAN:
            points = self.dt.median.values
        return points

    def _calc_err(self) -> Tuple[ndarray, ndarray] | None:
        """Calculate error bars for datapoints"""
        if self.err is _Err.STD:
            n_err = p_err = self.dt.std.values
        elif self.err is _Err.SEM:
            n_err = p_err = self.dt.sem.values
        elif self.err is _Err.CI:
            n_err = p_err = self._calc_ci()
        elif self.err is _Err.RANGE:
            avg = self._calc_avg()
            n_err, p_err = self.dt.max.values - avg, avg - self.dt.min.values
        else:  # NONE or ALL
            n_err, p_err = None, None
        return n_err, p_err

    def _calc_ci(self) -> ndarray:
        """Calculate confidence intervals"""
        from numpy import abs, all
        from scipy.stats import norm, t
        lt_probability = (1 - self.ci) / 2
        counts = self.dt.count.values
        if all(counts >= 30):
            # Use normal distribution approximation for n >= 30
            z_crit = abs(norm.ppf(lt_probability))
            ci = z_crit * self.dt.sem.values
        else:
            # Use t distribution for small sample size
            dof = counts - 1
            t_crit = abs(t.ppf(lt_probability, dof))
            ci = t_crit * self.dt.sem.values
        return ci

    def _err_plot(self, ax: Axes) -> None:
        """Generate plot with error bars"""
        points, (n_err, p_err) = self._calc_avg(), self._calc_err()
        # Transpose for convenient slicing
        points, n_err, p_err = points.T, n_err.T, p_err.T
        assert n_err is not None
        x, *x_err = points[0], n_err[0], p_err[0]
        y, *y_err = points[1:], n_err[1:], p_err[1:]
        if self.dt.n_x_replicates == 1:
            x_err = None
        plt_data = (self.dt.group_names, y, *y_err)
        # Assert the correct number of groups has been preserved
        assert len({len(f) for f in plt_data}) == 1
        for group_name, y, *y_err in zip(*plt_data):
            ax.errorbar(x, y, y_err, x_err,
                        label=group_name, **self.cfg["errorbar"])

    def _all_none_plot(self, ax: Axes) -> None:
        """Generate plot with no error bars or individual points plotted"""
        assert self.err is _Err.ALL or self.err is _Err.NONE
        points = self._calc_avg().T
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


class _Avg(Token):

    MEAN = 0
    MEDIAN = 1


class _Err(Token):

    STD = 0
    SEM = 1
    CI = 2
    RANGE = 3
    ALL = 4
    NONE = 5
