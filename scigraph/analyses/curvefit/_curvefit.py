from __future__ import annotations

__all__ = ["CurveFit"]

from abc import ABC, abstractmethod
import inspect
import itertools
from typing import Literal, NamedTuple, Optional, override, TYPE_CHECKING

from matplotlib.axes import Axes
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Index, MultiIndex
from scipy.optimize import curve_fit, root_scalar
import scipy.stats

from scigraph.analyses.abc import GraphableAnalysis
from scigraph._log import LOG
from scigraph._options import ConstraintType, CurveFitBands
from scigraph.config import SG_DEFAULTS

if TYPE_CHECKING:
    from scigraph.datatables import XYTable
    from scigraph.graphs import XYGraph


class CurveFit(GraphableAnalysis, ABC):

    def __init__(
        self,
        table: XYTable,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        confidence_level: float = 0.95,
    ) -> None:
        self._table = table

        # Fit attributes
        self._p0 = self._default_p0()
        self._upper_bounds = self._default_upper_bounds()
        self._lower_bounds = self._default_lower_bounds()
        self._bounds = self._lower_bounds, self._upper_bounds
        self._is_bound = self._default_is_bound()
        self._init_include_list(include, exclude)
        self._confidence_level = confidence_level

    @property
    @override
    def table(self) -> XYTable:
        return self._table

    @override
    def analyze(self) -> DataFrame:
        self.fit()
        return self.result

    @property
    @override
    def result(self) -> DataFrame:
        if hasattr(self, "_pretty_result"):
            return self._pretty_result

        data = []
        for r in self._result.values():
            fit_metrics = np.array([r.dof, r.r2, r.rss, r.sy_x, r.n])
            d = np.hstack([r.popt, r.ci, fit_metrics])
            data.append(d)

        data = np.vstack(data)
        index = [(category, param)
                 for category in ("Best Fit Params",
                                  f"CI {self._confidence_level:.0%}")
                 for param in self._params()]
        index.extend([("Goodness of Fit", stat)
                      for stat in ["dof", "r2", "rss", "sy_x", "n"]])
        index = MultiIndex.from_tuples(index)
        columns = Index(self._include)
        self._pretty_result = DataFrame(data.T, index, columns)

        return self._pretty_result

    def add_constraint(
        self,
        param: str,
        ty: Literal['equal', "greater", "less"],
        value: float,
        *,
        epsilon: float = 1e-8,
    ) -> None:
        i = self._get_param_index(param)

        constraint_t = ConstraintType.from_str(ty)
        match constraint_t:
            case ConstraintType.EQUAL:
                err = abs(value) * epsilon / 2
                self._upper_bounds[i] = value + err
                self._lower_bounds[i] = value - err
                self._p0[i] = value
                self._is_bound[i] = True
            case ConstraintType.GREATER:
                self._lower_bounds[i] = value
                if self._p0[i] < value:
                    self._p0[i] = value
            case ConstraintType.LESS:
                self._upper_bounds[i] = value
                if self._p0[i] > value:
                    self._p0[i] = value

    def add_initial_values(self, **values: float) -> None:
        for param, val in values.items():
            i = self._get_param_index(param)
            if not self._lower_bounds[i] <= val <= self._upper_bounds[i]:
                val = self._lower_bounds[i]
            self._p0[i] = val

    def fit(self) -> dict[str, CurveFitResult]:
        if self.table.n_x_replicates > 1:
            LOG.warn("Curve fit does not currently support multiple X values. "
                     "X values have been averaged.")

        x_ = self.table.x_values.mean(axis=1)

        assert x_.ndim == 1 and len(x_) == self.table.nrows
        x_ = np.tile(x_, self.table.n_y_replicates)

        result: dict[str, CurveFitResult] = {}

        for id, data in self.table.datasets_itertuples():
            if id not in self._include:
                continue

            y = data.y.T.flatten()
            nan_mask = ~np.isnan(y)
            x, y = x_[nan_mask], y[nan_mask]
            n = len(y)
            dof = n - self._n_params + self._is_bound.sum()
            assert len(x) == len(y)

            try:
                popt, pcov = curve_fit(self._f, x, y, p0=self._p0,
                                       bounds=self._bounds, nan_policy="omit")
            except RuntimeError as e:
                LOG.warn(f"Curve fit failed for {id}. SciPy error: {e}")
                null = np.full(self._n_params, np.nan)
                r = CurveFitResult(null, null, null, dof, np.nan, np.nan,
                                   np.nan, n, False)
                result[id] = r
                continue

            rss = ((y - self._f(x, *popt)) ** 2).sum()
            tss = ((y - y.mean()) ** 2).sum()
            r2 = 1 - (rss / tss)
            sy_x = (rss / dof) ** 0.5

            se = np.sqrt(np.diag(pcov))
            t = scipy.stats.t.ppf((1 + self._confidence_level) / 2, df=dof)
            ci = t * se

            result[id] = CurveFitResult(popt, pcov, ci, dof, r2, rss, sy_x, n,  # type: ignore
                                        True)

        self._result = result
        return self._result

    def predict(self, x: NDArray | ArrayLike, dataset: str) -> NDArray:
        self._access_check(dataset)
        popt = self._result[dataset].popt
        x = np.array(x)
        return self._f(x, *popt)

    def interpolate(
        self,
        y: NDArray | ArrayLike | float,
        dataset: str,
        n_steps: int = 1000,
        log_step: bool = False,
        xlims: Optional[tuple[float, float]] = None,
        **kwargs
    ) -> NDArray:
        if "_popt" in kwargs:
            # Allow spoofing of arbitrary popt values, allows GlobalCurveFit to
            # utilize this method, avoiding code duplication.
            popt = kwargs.pop("_popt")
        else:
            self._access_check(dataset)
            popt = self._result[dataset].popt

        if isinstance(y, float | int):
            y = [y]

        y = np.array(y)
        out = np.empty(y.size)

        if xlims is None:
            xlims = self._table.x_values.min(), self._table.x_values.max()

        xrange = xlims[1] - xlims[0]
        ext_lims = xlims[0] - xrange * 0.5, xlims[1] + xrange * 0.5

        bkp_f = np.logspace if log_step else np.linspace
        bkps = bkp_f(*xlims, n_steps)
        lower_ext_bkps = bkp_f(ext_lims[0], xlims[0], round(n_steps / 0.5))
        upper_ext_bkps = bkp_f(xlims[1], ext_lims[1], round(n_steps / 0.5))

        for i, v in enumerate(y):
            f = lambda x: self._f(x, *popt) - v

            def scan(bkps: NDArray) -> bool:
                for bracket in itertools.pairwise(bkps):
                    try:
                        sol = root_scalar(f, bracket=bracket, method="brentq",
                                          **kwargs)
                    except ValueError:
                        continue

                    if sol.converged:
                        out[i] = sol.root
                        return True
                return False
            
            if scan(bkps):
                continue

            LOG.warn(f"Failed to interpolate {v} in xrange {xlims}. "
                     f"Extending scan range to {ext_lims}")

            if scan(lower_ext_bkps) or scan(upper_ext_bkps):
                continue

            # Failed to find roots
            LOG.warn(f"Failed to interpolate {v} in extended xrange.")
            out[i] = np.nan

        out = out.reshape(y.shape)

        return out

    @override
    def draw(
        self,
        graph: XYGraph,
        ax: Axes,
        x_min: int | None = None,
        x_max: int | None = None,
        n_points: int = 1000,
        bands: Optional[Literal["confidence", "prediction"]] = None,
        line_kws: Optional[dict] = None,
        band_kws: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> None:
        if line_kws is None:
            line_kws = {}
        if band_kws is None:
            band_kws = {}

        # Determine x limits
        x = self.table.x_values.flatten()
        if graph.xaxis.scale == "log10":
            x = x[np.nonzero(x)]
        if x_min is None:
            x_min = x.min()
        if x_max is None:
            x_max = x.max()

        match graph.xaxis.scale:
            case "linear":
                xlim = x_min, x_max
                x = np.linspace(*xlim, n_points)  # type: ignore
            case "log10":
                xlim = np.log10(x_min), np.log10(x_max)  # type: ignore
                x = np.logspace(*xlim, n_points)
            case _:
                raise NotImplementedError

        for id, r in self._result.items():
            if not r.converged:
                continue

            props = graph.plot_properties[id]
            def_line_kws = props.line_kws()
            def_line_kws.update(**line_kws)

            y = self._f(x, *r.popt)
            line, = ax.plot(x, y, **def_line_kws)
            graph._add_legend_artist(id, line)

            if bands is None:
                continue

            def_band_kws = SG_DEFAULTS["analyses.curve_fit.bands"]
            def_band_kws["color"] = props.color
            def_band_kws.update(**band_kws)

            y = self._calculate_bands(x, id, CurveFitBands.from_str(bands))
            ax.fill_between(x, *y, **def_band_kws)

    def _calculate_bands(
        self,
        x: NDArray,
        dataset_id: str,
        bands: CurveFitBands,
    ) -> tuple[NDArray, NDArray]:
        """Use the delta method to calculate confidence bands."""
        r = self._result[dataset_id]
        y = self._f(x, *r.popt)

        # Calculate derivative gradients
        step = 1e-8
        glx = np.empty((len(x), len(r.popt)))

        for i, p in enumerate(r.popt):
            popt_ = np.copy(r.popt)
            gradients: list[NDArray] = []

            for sign in (1, -1):
                popt_[i] = (1 + step * sign) * p
                y_ = self._f(x, *popt_)
                dy = y_ - y
                dpopt = popt_[i] - p
                dydp = dy / dpopt
                gradients.append(dydp)

            g_up, g_down = gradients  # Use centered gradient
            glx[:, i] = (g_up + g_down) / 2

        c = np.sum(glx @ r.pcov * glx, axis=1)
        t = scipy.stats.t.ppf((1 + self._confidence_level) / 2, df=r.dof)

        if bands is CurveFitBands.CONFIDENCE:
            delta = np.sqrt(c) * t
        else:  # Prediction bands
            mse = r.rss / r.dof
            delta = np.sqrt(c + mse) * t

        return y - delta, y + delta

    def _default_p0(self) -> NDArray:
        return np.ones(self._n_params)

    def _default_upper_bounds(self) -> NDArray:
        return np.full(self._n_params, +np.inf)

    def _default_lower_bounds(self) -> NDArray:
        return np.full(self._n_params, -np.inf)

    def _default_is_bound(self) -> NDArray:
        return np.full(self._n_params, False)

    def _get_param_index(self, name: str) -> int:
        try:
            return self._params().index(name)
        except ValueError as e:
            raise ValueError(
                f"{name} is not a valid parameter. Valid options: "
                f"{", ".join(self._params())}"
            ) from e

    def _get_dataset_index(self, name: str) -> int:
        try:
            return self._table.dataset_ids.index(name)
        except ValueError as e:
            raise ValueError(
                f"{name} is not a valid dataset name. Valid options: "
                f"{", ".join(self._table.dataset_ids)}"
            ) from e

    def _get_include_index(self, name: str) -> int:
        try:
            return self._include.index(name)
        except ValueError as e:
            raise ValueError(
                f"{name} is not a valid dataset name. Valid options: "
                f"{", ".join(self._include)}"
            ) from e

    @staticmethod
    @abstractmethod
    def _f(x: NDArray, *args: float) -> NDArray: ...

    def _params(self) -> tuple[str, ...]:
        return tuple(inspect.signature(self._f).parameters.keys())[1:]

    @property
    def _n_params(self) -> int:
        return len(self._params())

    def _init_include_list(
        self,
        include: list[str] | None,
        exclude: list[str] | None,
    ) -> None:
        if include is not None and exclude is not None:
            raise ValueError("Include and exclude cannot both be specified.")

        if include is not None:
            self._include = include
        elif exclude is not None:
            self._include = [dataset for dataset in self._table.dataset_ids
                             if dataset not in exclude]
        else:
            self._include = self._table.dataset_ids

        self._n_included = len(self._include)

    def _access_check(self, dataset: Optional[str] = None) -> None:
        if not hasattr(self, "_result"):
            raise AttributeError("Curve has not been fitted. Call .fit()")

        if dataset is not None and dataset not in self._result:
            raise ValueError(f"{dataset} is not a valid dataset. Available "
                             f"datasets: {self.table.dataset_ids}")


class CurveFitResult(NamedTuple):
    popt: NDArray[np.float64]
    pcov: NDArray[np.float64]
    ci: NDArray[np.float64]
    dof: int
    r2: float
    rss: float
    sy_x: float
    n: int
    converged: bool
