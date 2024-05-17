from __future__ import annotations

__all__ = ["ExtraSumOfSquaresFTest"]

from dataclasses import dataclass
from typing import override, TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.stats

from scigraph.analyses.abc import Analysis
from scigraph._log import LOG

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from scigraph.analyses.curvefit import CurveFit
    from scigraph.datatables import XYTable


class ModelComparison(Analysis):

    def __init__(
        self,
        model_one: CurveFit,
        model_two: CurveFit,
    ) -> None:
        if not model_one.table is model_two.table:
            raise ValueError(
                "Curve fit analyses to be compared must be bound to the same "
                "XYTable."
            )

        if model_one._include != model_two._include:
            raise ValueError(
                "The datasets included in the analyses must be the same for "
                "both CurveFit analyses."
            )

        if not (model_one._fitted and model_two._fitted):
            raise ValueError("One or both of the models has not been fitted.")

        self._model_one = model_one
        self._model_two = model_two
        self._models = model_one, model_two
        self._compare_by_dataset = not (model_one._global_fit or model_two._global_fit)
        self._rss, self._dof, self._n = self._calculate_rss_dof_n()

    @property
    @override
    def table(self) -> XYTable:
        return self._model_one.table

    def _calculate_rss_dof_n(self) -> tuple[NDArray, NDArray, NDArray]:
        ncols = self._model_one._n_included if self._compare_by_dataset else 1
        rss = np.full((2, ncols), 0.0)
        dof = np.full((2, ncols), 0.0)
        n = np.full((2, ncols), 0.0)

        for i, model in enumerate(self._models):
            if model._global_fit:
                res = model._result[model._GLOBAL_NAME]
                if res is None:
                    LOG.warn(
                        f"CurveFit failed for {model._GLOBAL_NAME} and "
                        "comparison could not be made."
                    )
                    continue
                rss[i] = res.rss
                dof[i] = res.dof
                n[i] = res.n
                continue

            for j, (res_k, res) in enumerate(model._result.items()):
                if res is None:
                    LOG.warn(
                        f"CurveFit failed for {res_k} and has been excluded "
                        "from the comparison. Interpret results with caution."
                    )
                    continue
                if self._compare_by_dataset:
                    rss[i][j] = res.rss
                    dof[i][j] = res.dof
                    n[i][j] = res.n
                else:  # Aggregate rss and dof values across datasets
                    rss[i] += res.rss
                    dof[i] += res.dof
                    n[i] += res.n

        return rss, dof, n


class ExtraSumOfSquaresFTest(ModelComparison):

    def __init__(
        self,
        model_one: CurveFit,
        model_two: CurveFit,
    ) -> None:
        super().__init__(model_one, model_two)
        if not np.all(self._dof[0] > self._dof[1]):
            raise ValueError(
                "Model two does not have fewer degrees of freedom "
                f"({self._dof[1]}) than model one ({self._dof[0]}) and, so, is "
                "not more complex. The extra sum of squares f test is only "
                "valid for comparing nested models, where model two has fewer "
                "degrees of freedom. Swap model one and model two, if they "
                "have different degrees of freedom or consider using AIC "
                "Comparison if they have the same."
            )

    @property
    @override
    def table(self) -> XYTable:
        return self._model_one.table

    @override
    def analyze(self) -> pd.DataFrame:
        ss_1, ss_2 = self._rss
        df_1, df_2 = self._dof

        f = ((ss_1 - ss_2) / (df_1 - df_2)) / (ss_2 / df_2)
        p = scipy.stats.f.sf(f, df_1 - df_2, df_2)

        if self._compare_by_dataset:
            keys = self._model_one._include
        else:
            keys = [self._model_one._GLOBAL_NAME]

        values = [
            ExtraSumOfSquaresFTestResult(*vals)
            for vals in zip(ss_1, df_1, ss_2, df_2, f, p)
        ]
        self._result = dict(zip(keys, values))

        return pd.DataFrame(values, pd.Index(keys)).T


class AICComparison(ModelComparison):

    @override
    def analyze(self) -> pd.DataFrame:
        rss = self._rss
        n = self._n
        k = n - self._dof + 1
        aicc = 2 * k + n * np.log(rss / n) + (2 * k * (k + 1) / (n - k - 1))
        delta_aicc = aicc[0] - aicc[1]
        p2 = np.exp(0.5 * delta_aicc) / (1 + np.exp(0.5 * delta_aicc))
        p1 = 1 - p2
        ratio = p2 / p1

        if self._compare_by_dataset:
            keys = self._model_one._include
        else:
            keys = [self._model_one._GLOBAL_NAME]

        values = [AICComparisonResult(*vals) for vals in zip(delta_aicc, p1, p2, ratio)]
        self._result = dict(zip(keys, values))

        return pd.DataFrame(values, pd.Index(keys)).T


@dataclass(slots=True)
class ExtraSumOfSquaresFTestResult:
    ss_1: float
    df_1: float
    ss_2: float
    df_2: float
    f: float
    p: float


@dataclass(slots=True)
class AICComparisonResult:
    delta_aicc: float
    p1: float
    p2: float
    p_ratio: float
