from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import override

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from scipy.stats import t

from .abc import Graph
from ._formatters import (
    set_antilog_format,
    set_power10_format,
    set_log10_format
)
from ._options import (
    PtOpt,
    ErrOpt,
    ScaleOpt,
    Log10ScaleFmtOpt,
)
from scigraph.datatables.xy import XYTable
from scigraph._typing import PlottableXYAnalysis


class XYPointsGraph(Graph):

    def __init__(
        self,
        table: XYTable,
        pt_opt: PtOpt,
        err_opt: ErrOpt,
    ) -> None:
        self.table = table
        self._pt_opt = pt_opt
        self._err_opt = err_opt
        self._handles = {group: [] for group in self.table.ygroups}
        self.analyses: list[PlottableXYAnalysis] = []
        self.xaxis_opts = AxisOpts(label=table.xname)
        self.yaxis_opts = AxisOpts(label=table.yname)

    def link_analysis(
        self,
        analysis: PlottableXYAnalysis,
    ) -> None:
        if not isinstance(analysis, PlottableXYAnalysis):
            raise TypeError("Must be a PlottableXYAnalysis")
        if analysis.table is not self.table:
            raise ValueError("Linked analysis must be from the same DataTable")
        self.analyses.append(analysis)

    @override
    def draw(self, ax: plt.Axes = None) -> plt.Axes:
        if ax is None:
            ax = plt.gca()

        ax.set_xscale(self.xaxis_opts.scale.mpl_arg())
        ax.set_yscale(self.yaxis_opts.scale.mpl_arg())
        ax.set_xlabel(self.xaxis_opts.label)
        ax.set_ylabel(self.yaxis_opts.label)

        self._format_axis(ax.xaxis, self.xaxis_opts)
        self._format_axis(ax.yaxis, self.yaxis_opts)

        df = self.table.as_df()
        xpts = df.iloc[:, 0]

        if self.pt_opt == PtOpt.INDIVIDUAL:
            xpts = xpts.flatten()
            xpts = np.tile(self.table.n_replicates)

        for group, props in zip(
                self.table.ygroups, cycle(plt.rcParams["axes.prop_cycle"])):
            ydata = df[group].values
            ypts = self._prepare_ypts(ydata)
            assert len(xpts) == len(ypts)
            handler, = ax.plot(
                xpts,
                ypts,
                c=props["color"],
                marker=props["marker"],
                ls=""
            )
            if self._requires_errorbar():
                ax.errorbar(
                    xpts,
                    ypts,
                    xerr=self.table.xerr,
                    yerr=self._prepare_yerr(ydata, ypts),
                    c=props["color"],
                    marker="",
                    ls="",
                )
            self._handles[group].append(handler)

        for analysis in self.analyses:
            analysis.plot(ax, self)

        self._add_legend(ax)
        return ax

    def _add_legend(self, ax: plt.Axes):
        for k, v in self._handles.items():
            if len(v) == 1:
                self._handles[k] = v[0]
            else:
                self._handles[k] = tuple(v)
        return ax.legend(
            list(self._handles.values()),
            list(self._handles.keys()),
        )

    @property
    def pt_opt(self) -> PtOpt:
        return self._pt_opt

    @pt_opt.setter
    def pt_opt(self, val: PtOpt) -> None:
        if not isinstance(val, PtOpt):
            raise TypeError
        if not self._verify_opts(val, self.err_opt):
            return ValueError
        self._pt_opt = val

    @property
    def err_opt(self) -> ErrOpt:
        return self._err_opt

    @err_opt.setter
    def err_opt(self, val: ErrOpt) -> None:
        if not isinstance(val, ErrOpt):
            raise TypeError
        if not self._verify_opts(self.pt_opt, val):
            return ValueError
        self._err_opt = val

    def _verify_opts(self, pt_opt: PtOpt, err_opt: ErrOpt) -> bool:
        match pt_opt:
            case PtOpt.MEAN:
                valid = err_opt in [ErrOpt.SD, ErrOpt.SEM,
                                    ErrOpt.CI95, ErrOpt.NONE]
            case PtOpt.MEDIAN:
                valid = err_opt in [ErrOpt.RANGE, ErrOpt.IQR, ErrOpt.NONE]
            case PtOpt.INDIVIDUAL:
                err_opt == ErrOpt.NONE
        return valid

    def _prepare_ypts(self, ydata: NDArray) -> NDArray:
        match self.pt_opt:
            case PtOpt.MEAN:
                pts = np.mean(ydata, axis=1)
            case PtOpt.MEDIAN:
                pts = np.median(ydata, axis=1)
            case PtOpt.INDIVIDUAL:
                pts = np.transpose(ydata).flatten()
            case _:
                raise NotImplementedError
        return pts

    def _prepare_yerr(
        self,
        ydata: NDArray,
        ypts: NDArray
    ) -> NDArray | tuple[NDArray] | None:
        match self.err_opt:
            case ErrOpt.SD:
                yerr = np.std(ydata, axis=1)
            case ErrOpt.SEM:
                n = np.count_nonzero(~np.isnan(ydata), axis=1, keepdims=True)
                yerr = np.std(ydata, axis=1) / np.sqrt(n)
            case ErrOpt.CI95:
                n = np.count_nonzero(~np.isnan(ydata), axis=1, keepdims=True)
                critical_val = t.ppf(0.975, n - 1)
                yerr = critical_val * np.std(ydata, axis=1) / np.sqrt(n)
            case ErrOpt.RANGE:
                lower = ypts - np.min(ydata, axis=1)
                upper = np.max(ydata, axis=1) - ypts
                yerr = lower, upper
            case ErrOpt.IQR:
                lower = ypts - np.quantile(ydata, 0.25, axis=1)
                upper = np.quantile(ydata, 0.75, axis=1) - ypts
                yerr = lower, upper
            case ErrOpt.NONE:
                yerr = None
        return yerr

    def _requires_errorbar(self) -> bool:
        return self.err_opt != ErrOpt.NONE and self.pt_opt != PtOpt.INDIVIDUAL

    def _format_axis(self, axis: Axis, axis_opts: AxisOpts) -> None:
        match axis_opts.scale:
            case ScaleOpt.LOG10:
                match axis_opts.log10scalefmt:
                    case Log10ScaleFmtOpt.ANTILOG:
                        set_antilog_format(axis)
                    case Log10ScaleFmtOpt.POWER10:
                        set_power10_format(axis)
                    case Log10ScaleFmtOpt.LOG:
                        set_log10_format(axis)


@dataclass
class AxisOpts:
    label: str
    scale: ScaleOpt = ScaleOpt.default()
    log10scalefmt: Log10ScaleFmtOpt = Log10ScaleFmtOpt.default()
