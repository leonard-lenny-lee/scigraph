from __future__ import annotations

__all__ = ["GlobalCurveFit"]

from typing import TYPE_CHECKING, NamedTuple, Optional, Callable, Literal, override

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

    def __init__(self, model: CurveFit) -> None:
        self._model = model
        self._equal_params: list[str] = []
        self._p0 = np.tile(self._model._p0, self._model._n_included)
        self._bounds = self._compile_bounds()
        self._is_bound = np.tile(model._is_bound, self._model._n_included)
        self._is_constrained = np.full(self._model._n_params, False)

    def add_equality_constraint(self, param: str) -> None:
        i = self._model._get_param_index(param) 

        if param not in self._equal_params:
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

        if dataset is not None:
            ds_indices = [self._model._get_dataset_index(dataset)]
        else:  # Apply to all
            ds_indices = list(range(self._model._n_included))

        constraint_t = ConstraintType.from_str(ty)

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
        popt = r.popt.reshape(2, 3)
        shared = popt[0].copy()
        shared[~self._is_constrained] = np.nan
        popt = np.vstack((popt, shared)).T

        stats = np.full((4, popt.shape[1]), np.nan)
        stats[:, -1] = [r.dof, r.r2, r.rss, r.sy_x]
        values = np.vstack((popt, stats))

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

        result = minimize(obj, self._p0, args=xy, bounds=self._bounds,
                          constraints=constraints, method="SLSQP")

        if not result.success:
            LOG.warn("Global curve fit failed.")
            LOG.warn(f"SciPy message: {result.message}")
        
        y = self.table.y_values
        y = y[~np.isnan(y)]
        n = y.size
        dof = n - self._n_params + self._is_bound.sum() + self._is_constrained.sum()
        rss = result.fun
        tss = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (rss / tss)
        sy_x = (rss / dof) ** 0.5
        self._result = GlobalCurveFitResult(result.x, dof, r2, rss, sy_x)
        
        return self._result

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

    def _compile_constraints(self) -> NonlinearConstraint:
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

    def _compile_bounds(self) -> list[list[float]]:
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
