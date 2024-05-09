from __future__ import annotations

__all__ = ["CurveFit"]

from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
import itertools as it
import logging
import multiprocessing
from typing import Literal, Optional, override, TYPE_CHECKING
import warnings

from matplotlib.axes import Axes
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Index, MultiIndex
from scipy.optimize import NonlinearConstraint, curve_fit, minimize, root_scalar
import scipy.stats

from scigraph.analyses.abc import GraphableAnalysis
from scigraph._log import LOG
from scigraph._options import CFBoundType, CFBandType, CFReplicatePolicy
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
        *,
        replicate_policy: Literal["individual", "mean"] = "individual",
        confidence_level: float = 0.95,
    ) -> None:
        self._table = table
        self._init_include_list(include, exclude)
        self._n_params = len(self._params())
        self._replicate_policy = CFReplicatePolicy.from_str(replicate_policy)
        self._confidence_level = confidence_level

        # Fit attributes
        self._p0 = self._default_p0()
        self._upper_bounds = self._default_upper_bounds()
        self._lower_bounds = self._default_lower_bounds()
        self._bounds = self._lower_bounds, self._upper_bounds
        self._is_bound = self._default_is_bound()
        self._is_constrained = np.full(self._n_params, False)

        self._fitted = False

    @staticmethod
    @abstractmethod
    def _f(x: NDArray, *args: float) -> NDArray: ...

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

        self._check_fitted()
        data = []
        for r in self._result.values():
            if r is not None:
                gof_metrics = np.array([r.dof, r.r2, r.rss, r.sy_x, r.n])
                d = np.hstack([r.popt, gof_metrics])
                data.append(d)
            else:
                data.append(np.full(self._n_params + 5, np.nan))

        data = np.vstack(data)
        index = [
            ("Best Fit Params", param) for param in self._params()
        ]  # type: list[tuple[str, str]]
        index.extend(
            [("Goodness of Fit", stat) for stat in ["dof", "r2", "rss", "sy_x", "n"]]
        )
        self._pretty_result = DataFrame(
            data.T, MultiIndex.from_tuples(index), Index(self._result.keys())
        )

        return self._pretty_result

    def add_constraint(self, param: str) -> None:
        """Add an equality constraint to the minimization problem.

        Constrain a parameter so all the datasets must share the same value.

        Args:
            param: The name of the parameter to constrain.
        """
        # Check if parameter is also bound to a particular value
        i = self._get_param_index(param)
        idxs = list(range(i, self._total_n_params, self._n_params))

        if np.any(self._is_bound[idxs]):
            # Impose this restriction for now as it makes dof calculations more
            # difficult
            LOG.warn(
                "Equality constraint has been ignored as it is already "
                "bounded. A value cannot be bound and constrained "
                "simultaneously."
            )
            return

        self._is_constrained[i] = True

    def add_bound(
        self,
        param: str,
        ty: Literal["equal", "greater", "less"],
        value: float,
        dataset: Optional[str] = None,
        *,
        epsilon: float = 1e-8,
    ) -> None:
        i = self._get_param_index(param)
        constraint_t = CFBoundType.from_str(ty)

        if dataset is None:  # Apply bound to *all* datasets
            idxs = list(range(i, self._total_n_params, self._n_params))
        else:
            ds_idx = self._get_include_index(dataset)
            idxs = [i + ds_idx * self._n_params]

        if constraint_t is CFBoundType.EQUAL and np.any(self._is_constrained[i]):
            # Impose this restriction for now as it makes dof calculations more
            # difficult
            LOG.warn(
                "Equality bound has been ignored as it is already "
                "constrained. A value cannot be bound and constrained "
                "simulataneously."
            )
            return

        match constraint_t:
            case CFBoundType.EQUAL:
                err = abs(value) * epsilon / 2
                self._upper_bounds[idxs] = value + err
                self._lower_bounds[idxs] = value - err
                self._p0[idxs] = value
                self._is_bound[idxs] = True
            case CFBoundType.GREATER:
                self._lower_bounds[idxs] = value
                if self._p0[idxs] < value:
                    self._p0[idxs] = value
            case CFBoundType.LESS:
                self._upper_bounds[idxs] = value
                if self._p0[idxs] > value:
                    self._p0[idxs] = value

    def add_initial_value(
        self, param: str, val: float, dataset: Optional[str] = None
    ) -> None:
        i = self._get_include_index(param)

        if dataset is None:  # Apply to all
            idxs = list(range(i, self._total_n_params, self._n_params))
        else:
            ds_i = self._get_dataset_index(dataset)
            idxs = [ds_i * self._n_params + i]

        self._p0[idxs] = val

    def fit(self) -> dict[str, CurveFitResult | None]:
        use_global_fit = np.any(self._is_constrained)
        xy = self._prepare_xy()

        if use_global_fit:
            res = self._gfit(xy)
        else:
            res = self._fit(xy)

        self._fitted = True
        self._result = res
        return res

    def approximate_CI(self) -> DataFrame:
        self._check_fitted()
        n_datasets = len(self._result)
        lower_ci = np.full((n_datasets, self._n_params), np.nan)
        upper_ci = np.full((n_datasets, self._n_params), np.nan)

        for i, (id, res) in enumerate(self._result.items()):
            if res is None or res.pcov is None:
                LOG.warn(
                    f"Covariance could not be estimated for {id}. Try "
                    f"alternative methods to estimate confidence intervals."
                )
                continue
            se = np.sqrt(np.diag(res.pcov))
            t = scipy.stats.t.ppf((1 + self._confidence_level) / 2, df=res.dof)
            ci = t * se
            lower_ci[i] = res.popt - ci
            upper_ci[i] = res.popt + ci

        values = np.hstack((lower_ci, upper_ci))
        categories = (
            f"Lower CI {self._confidence_level:.0%}",
            f"Upper CI {self._confidence_level:.0%}",
        )
        index = MultiIndex.from_product((categories, self._params()))
        return DataFrame(values.T, index, Index(self._result.keys()))

    def bootstrap_CI(self, n_samples: int = 1000) -> DataFrame:
        """Estimate confidence intervals by subsampling with the bootstrap
        method.

        Use bootstraping to estimate the confidence intervals of the
        minimization. May be computationally expensive, but probably the best
        way to estimate confidence intervals.

        Args:
            n_samples: How many samples for bootstrapping.

        Returns:
            A summary dataframe of the confidence intervals obtained.
        """
        fn = _BootstrapSample(self)
        parameters = []
        chunksize = round(n_samples / multiprocessing.cpu_count())

        with multiprocessing.Pool() as p:
            for fvar in p.imap_unordered(fn, range(n_samples), chunksize=chunksize):
                parameters.append(fvar)

        parameters = np.vstack(parameters)

        cl_upper = (1 + self._confidence_level) / 2
        cl_lower = 1 - cl_upper
        ci = np.nanquantile(parameters, [cl_lower, 0.5, cl_upper], axis=0)

        for (ds, p_name), (lower, median, upper) in zip(
            it.product(self._include, self._params()), ci.T
        ):
            uncertainty = ((upper - median) + (median - lower)) / 2
            LOG.info(
                f"{ds} {p_name} = {median:.4g} ± {uncertainty:.4g}"
                f" ({lower:.4g} to {upper:.4g})"
            )

        # Reshape and reformat into DataFrame
        ci = ci.flatten("F").reshape(self._n_included, self._n_params * 3).T
        idx = (
            list(range(0, len(ci), 3))
            + list(range(1, len(ci), 3))
            + list(range(2, len(ci), 3))
        )
        ci = ci[idx]
        columns = self.table.dataset_ids.copy()
        used_global = np.any(self._is_constrained)

        if used_global:
            shared = ci.T[0].copy()
            shared[np.tile(~self._is_constrained, 3)] = np.nan
            ci = np.hstack((ci, np.atleast_2d(shared).T))
            columns.append("Global (Shared)")
            n_failures = np.isnan(parameters[:, 0]).sum()
        else:
            col_idx = list(range(0, self._total_n_params, self._n_params))
            n_failures = np.isnan(parameters[:, col_idx]).sum()

        if n_failures:
            failure_rate = n_failures / n_samples
            LOG.warn(
                f"Bootstrap minimization failures: {n_failures} / "
                f"{n_samples} ({failure_rate:.2%})"
            )

        index = MultiIndex.from_product((["Lower", "Median", "Upper"], self._params()))

        return DataFrame(ci, index, Index(columns))

    def predict(self, x: NDArray | ArrayLike, dataset: str) -> NDArray:
        popt = self._get_res(dataset).popt
        x = np.array(x)
        return self._f(x, *popt)

    def interpolate(
        self,
        y: NDArray | ArrayLike | float,
        dataset: str,
        n_steps: int = 1000,
        log_step: bool = False,
        xlims: Optional[tuple[float, float]] = None,
        **kwargs,
    ) -> NDArray:
        popt = self._get_res(dataset).popt

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
                for bracket in it.pairwise(bkps):
                    try:
                        sol = root_scalar(f, bracket=bracket, method="brentq", **kwargs)
                    except ValueError:
                        continue

                    if sol.converged:
                        out[i] = sol.root
                        return True
                return False

            if scan(bkps):
                continue

            LOG.warn(
                f"Failed to interpolate {v} in xrange {xlims}. "
                f"Extending scan range to {ext_lims}."
            )

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
            if id not in self._include:
                continue
            if r is None:
                continue

            props = graph.plot_properties[id]
            def_line_kws = props.line_kws()
            def_line_kws.update(**line_kws)

            y = self._f(x, *r.popt)
            (line,) = ax.plot(x, y, **def_line_kws)
            graph._add_legend_artist(id, line)

            if bands is None:
                continue

            y = self._calculate_bands(x, id, CFBandType.from_str(bands))
            if y is None:
                continue

            def_band_kws = SG_DEFAULTS["analyses.curve_fit.bands"]
            def_band_kws["color"] = props.color
            def_band_kws.update(**band_kws)
            ax.fill_between(x, *y, **def_band_kws)

    def _prepare_xy(self, subsample: bool = False) -> tuple[NDArray, ...]:
        """Organize the values into tuple of x and y arrays, arranged as:
        x1, y1, x2, y2, ... xn, yn; where n is the number of included datasets,
        and all the NaN values have been trimmed.
        """
        if self.table.n_x_replicates > 1:
            LOG.warn(
                "Curve fit does not currently support multiple X values. "
                "X values have been averaged."
            )

        x = self.table.x_values.mean(axis=1)
        out = []

        for id in self._include:
            ds = self.table.get_dataset(id)
            if self._replicate_policy is CFReplicatePolicy.INDIVIDUAL:
                x_ = np.tile(x, self.table.n_y_replicates)
                y_ = ds.y.flatten("F")
            else:  # Mean
                x_ = x
                y_ = ds.y.mean(axis=1)
            assert len(x_) == len(y_)
            not_nan = ~np.isnan(y_)
            x_, y_ = x_[not_nan], y_[not_nan]
            if subsample:
                idx = np.random.choice(len(x_), size=len(x_), replace=True)
                x_, y_ = x_[idx], y_[idx]
            out.append(x_)
            out.append(y_)

        return tuple(out)

    def _fit(self, xy: tuple[NDArray, ...]) -> dict[str, CurveFitResult | None]:
        """Fit each dataset to the model separately"""
        res: dict[str, CurveFitResult | None] = {}

        for i, (id, (x, y)) in enumerate(zip(self._include, it.batched(xy, 2))):
            s = slice(i * self._n_params, (i + 1) * self._n_params)
            p0 = self._p0[s]
            bounds = self._lower_bounds[s], self._upper_bounds[s]

            try:
                with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
                    popt, pcov = curve_fit(
                        self._f, x, y, p0=p0, bounds=bounds, nan_policy="omit"
                    )
            except RuntimeError as e:
                LOG.warn(f"Curve fit failed for {id}. SciPy error: {e}")
                res[id] = None
                continue

            # Goodness of fit parameters
            n = len(y)
            dof = n - self._n_params + self._is_bound[s].sum()
            rss = ((y - self._f(x, *popt)) ** 2).sum()
            tss = ((y - y.mean()) ** 2).sum()
            r2 = 1 - (rss / tss)
            sy_x = (rss / dof) ** 0.5

            res[id] = CurveFitResult(popt, pcov, dof, r2, rss, sy_x, n)

        return res

    def _gfit(self, xy: tuple[NDArray, ...]) -> dict[str, CurveFitResult | None]:
        """Fit all the datasets at once. Use optimize.minimize with SLSQP
        method as it allows for constrained minimization problems.
        """
        bounds = self._gbounds()
        constr = self._gconstr()

        with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
            r = minimize(
                self._gcost,
                self._p0,
                args=xy,
                bounds=bounds,
                constraints=constr,
                method="SLSQP",
            )

        if not r.success:
            LOG.warn("Global curve fit failed.")
            LOG.warn(f"SciPy minimzation error: {r.message}")

        popts = r.x.reshape(self._n_included, self._n_params)
        res = {}

        for popt, id, (x, y) in zip(popts, self._include, it.batched(xy, 2)):
            n = len(y)
            rss = ((y - self._f(x, *popt))).sum()
            tss = ((y - y.mean()) ** 2).sum()
            r2 = 1 - (rss / tss)
            res[id] = CurveFitResult(
                popt=popt,
                pcov=None,
                dof=np.nan,
                r2=r2,
                rss=rss,
                sy_x=np.nan,
                n=n,
            )

        # Global results
        y = self.table.y_values.flatten()
        y = y[~np.isnan(y)]
        n = y.size

        popt = popts[0].copy()
        popt[~self._is_constrained] = np.nan
        # Assertion must pass for dof calculation to be valid
        assert all(~(np.tile(self._is_constrained, self._n_included) & self._is_bound))
        dof = (
            n - self._total_n_params + self._is_bound.sum() + self._is_constrained.sum()
        )

        rss = r.fun
        tss = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (rss / tss)
        sy_x = (rss / dof) ** 0.5
        res["Global (Shared)"] = CurveFitResult(
            popt,
            pcov=None,
            dof=dof,
            r2=r2,
            rss=rss,
            sy_x=sy_x,
            n=n,
        )

        return res

    def _gcost(self, params: NDArray, *xy: NDArray) -> float:
        """Global cost function for optimize.minimize."""
        # Params are organized into blocks, in the order of the dataset
        # to be fit e.g. a1, b1, c1, a2, b2, c2, ...
        # Likewise, x and y are organized into blocks x1, y1, x2, y2, ...
        rss = 0

        for i in range(self._n_included):
            x, y = xy[i * 2 : i * 2 + 2]
            p = params[i * self._n_params : (i + 1) * self._n_params]
            y_pred = self._f(x, *p)
            rss += ((y - y_pred) ** 2).sum()

        return rss

    def _gconstr(self) -> NonlinearConstraint | None:
        """Generate the Constraint object to pass onto optimize.minimize"""
        if not np.any(self._is_constrained):
            return None

        def f(x):
            constraints = []
            for i in np.flatnonzero(self._is_constrained):
                for n in range(1, self._n_included):
                    constraint = x[i] - x[i + n * self._n_params]
                    constraints.append(constraint)
            return constraints

        bound = np.zeros(self._n_included - 1)
        return NonlinearConstraint(f, bound, bound)

    def _gbounds(self) -> list[list[float]]:
        """Reshape the bounds into a format for optimize.minimize"""
        return [list(bounds) for bounds in zip(*self._bounds)]

    def _calculate_bands(
        self,
        x: NDArray,
        dataset_id: str,
        bands: CFBandType,
    ) -> tuple[NDArray, NDArray] | None:
        """Use the delta method to calculate confidence bands."""
        r = self._result[dataset_id]
        if r is None or r.pcov is None:
            LOG.warn(
                f"Bands could not be plot as covariance could not be estimated "
                f"for {dataset_id}."
            )
            return

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

        if bands is CFBandType.CONFIDENCE:
            delta = np.sqrt(c) * t
        else:  # Prediction bands
            mse = r.rss / r.dof
            delta = np.sqrt(c + mse) * t

        return y - delta, y + delta

    def _default_p0(self) -> NDArray:
        """Override in child classes if rules can be defined."""
        return np.ones(self._total_n_params)

    def _default_upper_bounds(self) -> NDArray:
        return np.full(self._total_n_params, +np.inf)

    def _default_lower_bounds(self) -> NDArray:
        return np.full(self._total_n_params, -np.inf)

    def _default_is_bound(self) -> NDArray:
        return np.full(self._total_n_params, False)

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

    def _params(self) -> tuple[str, ...]:
        return tuple(inspect.signature(self._f).parameters.keys())[1:]

    @property
    def _total_n_params(self) -> int:
        return self._n_params * self._n_included

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
            self._include = [
                dataset for dataset in self._table.dataset_ids if dataset not in exclude
            ]
        else:
            self._include = self._table.dataset_ids

        self._n_included = len(self._include)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise AttributeError("Curve has not been fitted. Call .fit()")

    def _access_check(self, dataset: Optional[str] = None) -> None:
        self._check_fitted()

        if dataset is not None and dataset not in self._result:
            raise ValueError(
                f"{dataset} is not a valid dataset. Available "
                f"datasets: {self.table.dataset_ids}"
            )

    def _get_res(self, dataset: str) -> CurveFitResult:
        self._access_check(dataset)
        res = self._result[dataset]
        if res is None:
            raise ValueError(f"Curve fit failed for {dataset}.")
        return res


@dataclass(frozen=True, slots=True)
class CurveFitResult:
    popt: NDArray[np.float64]
    pcov: NDArray[np.float64] | None
    dof: int | float
    r2: float
    rss: float
    sy_x: float
    n: int


class _BootstrapSample:
    """Callable for a single bootstrap sample. Wrapped like this to enable the
    CurveFit object to be shared between processes.
    """

    def __init__(self, cf: CurveFit) -> None:
        self.cf = cf

    def __call__(self, *_) -> NDArray:
        logging.disable(logging.CRITICAL)
        xy = self.cf._prepare_xy(subsample=True)
        use_global_fit = np.any(self.cf._is_constrained)
        out = np.full((self.cf._n_included, self.cf._n_params), np.nan)

        if use_global_fit:
            res = self.cf._gfit(xy)
        else:
            res = self.cf._fit(xy)

        for i, id in enumerate(self.cf._include):
            r = res[id]
            if r is None:
                continue
            out[i] = r.popt

        return out.flatten()
