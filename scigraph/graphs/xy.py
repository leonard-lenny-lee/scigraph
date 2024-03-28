from enum import Enum
from typing import override

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import t

from .abc import Graph
from scigraph.datatables.xy import XYTable
import scigraph.styles as ss
from scigraph._typing import PlottableXYAnalysis


class PtOpt(Enum):
    MEAN = 0
    MEDIAN = 1
    INDIVIDUAL = 2


class ErrOpt(Enum):
    SD = 0
    SEM = 1
    CI95 = 2
    RANGE = 3
    IQR = 4
    NONE = 5


class ScaleOpt(Enum):
    LINEAR = 0
    LOG = 1

    def mpl_arg(self) -> str:
        match self:
            case self.LINEAR:
                out = "linear"
            case self.LOG:
                out = "log"
            case _:
                raise NotImplementedError
        return out


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
        self.xscale = ScaleOpt.LINEAR
        self.yscale = ScaleOpt.LINEAR
        self.xlabel = table.xname
        self.ylabel = table.yname

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
        ss.use("default")
        if ax is None:
            ax = plt.gca()
        ax.set_xscale(self.xscale.mpl_arg())
        ax.set_yscale(self.yscale.mpl_arg())
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        df = self.table.as_df()
        xpts = df.iloc[:, 0]
        if self.pt_opt == PtOpt.INDIVIDUAL:
            xpts = xpts.flatten()
            xpts = np.tile(self.table.n_replicates)
        with ss.context(ss.BASE_SS, after_reset=True):
            for group in self.table.ygroups:
                ydata = df[group].values
                ypts = self._prepare_ypts(ydata)
                assert len(xpts) == len(ypts)
                if self.pt_opt == PtOpt.INDIVIDUAL \
                        or self.err_opt == ErrOpt.NONE:
                    handler = ax.plot(xpts, ypts, linestyle="")
                else:
                    handler = ax.errorbar(
                        xpts,
                        ypts,
                        xerr=self.table.xerr,
                        yerr=self._prepare_yerr(ydata, ypts),
                        linestyle="",
                    )
                self._handles[group].append(handler)
            for analysis in self.analyses:
                analysis.plot(ax, self)
        self._add_legend(ax)
        return ax

    def _add_legend(self, ax: plt.Axes):
        for k, v in self._handles.items():
            self._handles[k] = tuple(v)
        return ax.legend(
            handles=list(self._handles.values()),
            labels=list(self._handles.keys()),
            handler_map={tuple: HandlerTuple},
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
