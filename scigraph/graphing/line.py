"""Contains the LineGraph class.
"""
from __future__ import annotations
from typing import Tuple
import warnings
import matplotlib.pyplot as plt
from numpy import ndarray, tile
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
    ):
        self.dt = dt
        self.avg = _Avg.from_str(avg)
        self.err = _Err.from_str(err)
        self.ci = ci

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
    def dt(self) -> XYTable:
        return self._dt

    @dt.setter
    def dt(self, dt: XYTable) -> None:
        if not isinstance(dt, XYTable):
            raise ValueError(f"Only XYTables can be used with LineGraph")
        self._dt = dt

    @property
    def err(self):
        return self._err

    @err.setter
    def err(self, err: _Err) -> None:
        """Emit warnings if specified errors are incompatible with data"""
        if self.dt.n_y_replicates == 1 and self.dt.n_x_replicates == 1:
            warnings.warn(
                f"DataTable contains no replicates to plot error with "
                f"method '{err}'. Defaulting to 'none'. ",
                RuntimeWarning
            )
            err = _Err.NONE
        if self.dt.n_x_replicates > 1 and err is _Err.ALL:
            warnings.warn(
                "Scigraph unable to plot more than one x replicate. "
                "Plotting x values as an average of all x replicates. ",
                RuntimeWarning
            )
        self._err = err

    @property
    def _grouped(self):
        return self.dt.data.groupby(axis=1, level=0, sort=False)

    @property
    def _std(self) -> ndarray:
        return self._grouped.std(numeric_only=True).values

    @property
    def _min(self) -> ndarray:
        return self._grouped.min().values

    @property
    def _max(self) -> ndarray:
        return self._grouped.max().values

    @property
    def _count(self) -> ndarray:
        return self._grouped.count().values

    @property
    def _sem(self) -> ndarray:
        return self._std / (self._count ** 0.5)

    def _calc_avg(self) -> ndarray:
        """Calculate XY data points"""
        if self.avg is _Avg.MEAN:
            points = self._grouped.mean().values
        elif self.avg is _Avg.MEDIAN:
            points = self._grouped.median().values
        return points.T

    def _calc_err(self) -> Tuple[ndarray, ndarray] | None:
        """Calculate error bars for datapoints"""
        if self.err is _Err.STD:
            n_err = p_err = self._std
        elif self.err is _Err.SEM:
            n_err = p_err = self._sem
        elif self.err is _Err.CI:
            n_err = p_err = self._ci()
        elif self.err is _Err.RANGE:
            avg = self._calc_avg()
            n_err, p_err = self._max - avg, avg - self._min
        else:  # NONE or ALL
            return None, None
        return n_err.T, p_err.T

    def _ci(self) -> ndarray:
        """Calculate confidence intervals"""
        from numpy import abs, all
        from scipy.stats import norm, t
        assert 0 < self.ci < 1
        lt_probability = (1 - self.ci) / 2
        if all(self._count >= 30):
            # Use normal distribution approximation for n >= 30
            z_crit = abs(norm.ppf(lt_probability))
            return z_crit * self._sem
        else:
            # Use t distribution for small sample size
            dof = self._count - 1
            t_crit = abs(t.ppf(lt_probability, dof))
            return t_crit * self._sem

    def _err_plot(self, fig_kw):
        """Generate plot with error bars"""
        fig, ax = plt.subplots(**fig_kw)
        points, (n_err, p_err) = self._calc_avg(), self._calc_err()
        assert n_err is not None
        x, *x_err = points[0], n_err[0], p_err[0]
        y, y_n_err, y_p_err = points[1:], n_err[1:], p_err[1:]
        plt_data = (self.dt.group_names, y, y_n_err, y_p_err)
        # Assert the correct number of groups has been preserved
        assert len({len(f) for f in plt_data}) == 1
        if self.dt.n_x_replicates == 1:
            x_err = None
        for group_name, y, *y_err in zip(*plt_data):
            ax.plot(x, y, label=group_name)
            ax.errorbar(x, y, y_err, x_err, fmt="none")
        return fig, ax

    def _all_none_plot(self, fig_kw):
        """Generate plot with no error bars or individual points plotted"""
        assert self.err is _Err.ALL or self.err is _Err.NONE
        fig, ax = plt.subplots(**fig_kw)
        points = self._calc_avg()
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
