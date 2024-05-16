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


class ExtraSumOfSquaresFTest(Analysis):

    def __init__(
        self,
        model_one: CurveFit,
        model_two: CurveFit,  # Complex model
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
        self._rss, self._dof = self._calculate_sum_of_squares_and_dof()

        if not np.all(self._dof[0] > self._dof[1]):
            raise ValueError(
                "Model two does not have fewer degrees of freedom "
                f"({self._dof[1]}) than model one ({self._dof[0]}) and, so, is "
                "not more complex. The extra sum of squares f test is only "
                "valid for comparing nested models. Consider AIC comparison."
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

    def _calculate_sum_of_squares_and_dof(self) -> tuple[NDArray, NDArray]:
        ncols = self._model_one._n_included if self._compare_by_dataset else 1
        rss = np.full((2, ncols), 0.0)
        dof = np.full((2, ncols), 0.0)

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
                else:  # Aggregate rss and dof values across datasets
                    if res.rss != np.nan:
                        rss[i] += res.rss
                    if res.dof != np.nan:
                        dof[i] += res.dof

        return rss, dof


class AICComparison(Analysis):

    def __init__(
        self,
        model_one: CurveFit,
        model_two: CurveFit,
    ) -> None:
        if not model_one.table is model_two.table:
            raise ValueError()

        self._model_one = model_one
        self._model_two = model_two

    @property
    @override
    def table(self) -> XYTable: ...

    def analyze(self) -> AICComparisonResult: ...


@dataclass(slots=True)
class ExtraSumOfSquaresFTestResult:
    ss_1: float
    df_1: float
    ss_2: float
    df_2: float
    f: float
    p: float


@dataclass(slots=True)
class AICComparisonResult: ...
