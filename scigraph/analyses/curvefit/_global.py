from __future__ import annotations

__all__ = ["GlobalCurveFit"]

import itertools as it
import multiprocessing as mp
from typing import TYPE_CHECKING, NamedTuple, Optional, Callable, Literal, override
import warnings

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Index, MultiIndex
from scipy.optimize import minimize, NonlinearConstraint

from scigraph.analyses.abc import Analysis
from scigraph._log import LOG
from scigraph._options import ConstraintType

if TYPE_CHECKING:
    from scigraph.analyses.curvefit import CurveFit
    from scigraph.datatables import XYTable


class GlobalCurveFit(Analysis):

    def __init__(
        self,
        model: CurveFit,
        inherit: bool = True,
    ) -> None:
        self._model = model
        self._inherit = inherit
        self._equal_params: list[str] = []
        self._p0 = self._compile_p0()
        self._bounds = self._compile_bounds()
        self._is_bound = np.tile(model._is_bound, self._model._n_included)
        self._is_constrained = np.full(self._model._n_params, False)

    def add_equality_constraint(self, param: str) -> None:
        """Add an equality constraint to the minimization problem.

        Constrain a parameter so all the datasets must share the same value.

        Args:
            param: The name of the parameter to constrain.
        """
        if param in self._equal_params:
            return

        i = self._model._get_param_index(param)

        if any([self._is_bound[ds_i * self._model._n_params + i]
                for ds_i in range(self._model._n_included)]):
            LOG.warn("Equality constraint has been ignored as it is already "
                     "bounded. A value cannot be bound and constrained "
                     "simultaneously.")
            return

        self._equal_params.append(param)
        self._is_constrained[i] = True

    def add_initial_value(self, param: str, val: float,
                          dataset: Optional[str] = None) -> None:
        p_i = self._model._get_param_index(param)

        if dataset is not None:
            ds_indices = [self._model._get_dataset_index(dataset)]
        else:  # Apply to all
            ds_indices = list(range(self._model._n_included))

        for ds_i in ds_indices:
            i = ds_i * self._model._n_params + p_i
            self._p0[i] = val

    def add_bound_constraint(
        self,
        param: str,
        ty: Literal['equal', "greater", "less"],
        value: float,
        dataset: Optional[str] = None,
        *,
        epsilon: float = 1e-8,
    ) -> None:
        p_i = self._model._get_param_index(param)
        constraint_t = ConstraintType.from_str(ty)

        if constraint_t is ConstraintType.EQUAL and self._is_constrained[p_i]:
            LOG.warn("Equality bound has been ignored as it is already "
                     "constrained. A value cannot be bound and constrained "
                     "simulataneously.")
            return

        if dataset is not None:
            ds_indices = [self._model._get_dataset_index(dataset)]
        else:  # Apply to all
            ds_indices = list(range(self._model._n_included))

        for ds_i in ds_indices:
            i = ds_i * self._model._n_params + p_i
            match constraint_t:
                case ConstraintType.EQUAL:
                    err = abs(value) * epsilon / 2
                    self._bounds[i][0] = value - err
                    self._bounds[i][1] = value + err
                    self._is_bound[i] = True
                    self._p0[i] = value
                case ConstraintType.GREATER:
                    self._bounds[i][0] = value
                    if self._p0[i] < value:
                        self._p0[i] = value
                case ConstraintType.LESS:
                    self._bounds[i][1] = value
                    if self._p0[i] > value:
                        self._p0[i] = value

    @property
    @override
    def table(self) -> XYTable:
        return self._model._table

    @override
    def analyze(self) -> DataFrame:
        r = self.fit()

        # Format into DataFrame for nice rendering
        values = []
        popt = r.popt.reshape(self._model._n_included, self._model._n_params)
        shared = popt[0].copy()
        shared[~self._is_constrained] = np.nan
        popt = np.vstack((popt, shared)).T
        values.append(popt)

        stats = np.full((4, popt.shape[1]), np.nan)
        stats[:, -1] = [r.dof, r.r2, r.rss, r.sy_x]
        values.append(stats)

        values = np.vstack(values)

        columns = self.table.dataset_ids.copy()
        columns.append("Global (Shared)")

        index = []
        index.extend([("Best Fit Params", p) for p in self._model._params()])
        index.extend([("Goodness of Fit", name)
                      for name in ["dof", "r2", "rss", "sy_x"]])

        return DataFrame(values, MultiIndex.from_tuples(index), Index(columns))

    def fit(self) -> GlobalCurveFitResult:
        xy = self._compile_xy()
        obj = self._compile_cost_fn()
        constraints = self._compile_constraints()

        with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
            result = minimize(obj, self._p0, args=xy, bounds=self._bounds,
                              constraints=constraints, method="SLSQP")

        if not result.success:
            LOG.warn("Global curve fit failed.")
            LOG.warn(f"SciPy minimzation error: {result.message}")

        y = self.table.y_values
        y = y[~np.isnan(y)]
        n = y.size

        # Assertion is required to ensure dof calculation is valid
        assert all(~(np.tile(self._is_constrained, self._model._n_included) & self._is_bound))  
        dof = n - self._n_params + self._is_bound.sum() + self._is_constrained.sum()

        rss = result.fun
        tss = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (rss / tss)
        sy_x = (rss / dof) ** 0.5
        self._result = GlobalCurveFitResult(result.x, dof, r2, rss, sy_x)
        self._scipy_result = result
        
        return self._result

    def estimate_confidence_intervals(
        self,
        confidence_level: float = 0.95,
        n_samples: int = 1000,
        sampling_ratio: float = 1.0,
    ) -> DataFrame:
        """Estimate confidence intervals with the bootstrapping method.

        Use bootstraping to estimate the confidence intervals of the
        minimization. May be computationally expensive, but probably the best
        way to estimate confidence intervals.

        Args:
            confidence_level: The confidence level of the intervals.
            n_samples: How many samples for bootstrapping.
            sampling_ratio: The proportion of points for each sample to take.

        Returns:
            A summary dataframe of the confidence intervals obtained.
        """
        xy = self._compile_xy()
        fn = _BootstrapSample(self, sampling_ratio, xy)
        parameters = []
        chunksize = round(n_samples / mp.cpu_count())

        with mp.Pool() as p:
            for fvar in p.imap_unordered(fn, range(n_samples),
                                         chunksize=chunksize):
                parameters.append(fvar)

        parameters = np.vstack(parameters)
        n_failures = np.isnan(parameters[:, 0]).sum()

        if n_failures:
            failure_rate = n_failures / n_samples
            LOG.warn(f"Bootstrap minimization failures: {n_failures} / "
                     f"{n_samples} ({failure_rate:.2%})")

        cl_upper = (1 + confidence_level) / 2
        cl_lower = 1 - cl_upper
        ci = np.nanquantile(parameters, [cl_lower, cl_upper],
                                              axis=0)

        for (ds, p_name), p, (l, u) in zip(
            it.product(self._model._include, self._model._params()),
            self.result.popt, ci.T
        ):
            uncertainty = ((u - p) + (p - l)) / 2
            LOG.info(f"{ds} {p_name} = {p:.4g} ± {uncertainty:.4g}")

        # Reshape and reformat into DataFrame
        ci = ci.flatten('F').reshape(self._model._n_included,
                                     self._model._n_params * 2).T
        idx = list(range(0, len(ci), 2)) + list(range(1, len(ci), 2))
        ci = ci[idx]

        shared = ci.T[0].copy()
        shared[np.tile(~self._is_constrained, 2)] = np.nan
        ci = np.hstack((ci, np.atleast_2d(shared).T))

        columns = self.table.dataset_ids.copy()
        columns.append("Global (Shared)")
        index = MultiIndex.from_product((["Lower" , "Upper"],
                                         self._model._params()))

        return DataFrame(ci, index, Index(columns)) 

    def _compile_xy(self) -> tuple[NDArray, ...]:
        """Return the x and y arrays to pass onto the cost function.
        Organized as: x1, y1, x2, y2, ...
        """
        if self.table.n_x_replicates > 1:
            LOG.warn("Curve fit does not currently support multiple X values. "
                     "X values have been averaged.")

        x = self.table.x_values.mean(axis=1)
        x = np.tile(x, self.table.n_y_replicates)
        out = []

        for id in self._model._include:
            ds = self.table.get_dataset(id)
            y = ds.y.flatten('F')
            assert len(x) == len(y)
            nan_mask = ~np.isnan(y)
            x_, y_ = x[nan_mask], y[nan_mask]
            out.append(x_)
            out.append(y_)

        return tuple(out)

    def _compile_cost_fn(self) -> Callable:
        """Return the cost function for optimize.minimize. Use RSS."""
        n_params = self._model._n_params

        def obj(params: NDArray, *xy: NDArray) -> float:
            # Params are organized into blocks, in the order of the dataset
            # to be fit e.g. a1, b1, c1, a2, b2, c2, ...
            # Likewise, x and y are organized into blocks x1, y1, x2, y2, ...
            rss = 0

            for i in range(self._model._n_included):
                x, y = xy[i*2:i*2+2]
                p = params[i*n_params:(i+1)*n_params]
                y_pred = self._model._f(x, *p)
                rss += ((y - y_pred) ** 2).sum()

            return rss

        return obj

    def _compile_constraints(self) -> NonlinearConstraint | None:
        if not self._equal_params:
            return None

        n_params = self._model._n_params
        bounds = np.zeros(self.table.n_datasets - 1)

        def constraint_f(params):
            out = []
            for c in self._equal_params:
                i = self._model._get_param_index(c)
                for n in range(1, self.table.n_datasets):
                    constraint = params[i] - params[i+n*n_params]
                    out.append(constraint)
            return out

        constraint = NonlinearConstraint(constraint_f, bounds, bounds)
        return constraint

    def _compile_p0(self) -> NDArray:
        if self._inherit:
            p0 = np.tile(self._model._p0, self._model._n_included)
        else:
            p0 = np.ones(self._n_params)
        return p0

    def _compile_bounds(self) -> list[list[float]]:
        if not self._inherit:
            return [[-np.inf, +np.inf] for _ in range(self.table.n_datasets)]

        bounds = []

        for _ in range(self.table.n_datasets):
            for p in self._model._params():
                i = self._model._get_param_index(p)
                upper = self._model._upper_bounds[i]
                lower = self._model._lower_bounds[i]
                bounds.append([lower, upper])

        return bounds

    @property
    def _n_params(self) -> int:
        return self._model._n_params * self._model._n_included


class GlobalCurveFitResult(NamedTuple):
    popt: NDArray
    dof: int
    r2: float
    rss: float
    sy_x: float


class _BootstrapSample:
    """Callable for a single bootstrap sample. Wrapped like this to enable the
    GlobalCurveFit object to be shared between processes.
    """

    def __init__(self, cf: GlobalCurveFit, sampling_ratio: float,
                 xy: tuple[NDArray, ...]) -> None:
        self.cf = cf
        self.xy = xy
        self.sampling_ratio = sampling_ratio

    def __call__(self, *_) -> NDArray:
        # Not picklable so have to create in each call
        obj = self.cf._compile_cost_fn()
        constr = self.cf._compile_constraints()

        bootstrap_data = []

        for x, y in it.batched(self.xy, 2):
            size = round(len(x) * self.sampling_ratio)
            idx = np.random.choice(len(x), size=size, replace=True)
            x, y = x[idx], y[idx]
            bootstrap_data.append(x)
            bootstrap_data.append(y)

        bootstrap_data = tuple(bootstrap_data)

        with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
            result = minimize(obj, self.cf._p0, args=bootstrap_data,
                              bounds=self.cf._bounds, constraints=constr,
                              method="SLSQP")
        
        if not result.success:
            r = np.full(self.cf._n_params, np.nan)
        else:
            r = result.x

        return r
