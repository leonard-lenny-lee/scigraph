from __future__ import annotations

__all__ = ["compare_models"]

from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, override, TYPE_CHECKING

import numpy as np
import pandas as pd
from prettytable import PrettyTable
import scipy.stats

from scigraph.analyses.abc import Analysis
from scigraph._log import LOG
from scigraph._options import CFComparisonMethod

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from scigraph.analyses.curvefit import CurveFit
    from scigraph.datatables import XYTable


def compare_models(
    model_one: CurveFit,
    model_two: CurveFit,
    method: Literal["f", "aic"],
    alpha: float = 0.05,
) -> dict[str, ExtraSumOfSquaresFTestResult] | dict[str, AICComparisonResult]:
    """For each dataset, which model fits best?

    Compare which model fits the data best using an extra sum-of-squares F test
    or Akaike's Information Criterion. If the F-test is chosen, the models must
    be nested where one model is a special case of the other. If models are not
    nested, choose AIC comparison.

    Args:
        model_one: The first model.
        model_two: The second model.
        method: The method to use to compare; "f" or "aic".
        alpha: The threshold p-value to select model two in the F-test analysis,
            ignored for AIC comparisons

    Returns:
        The model comparisons for each dataset.
    """
    cmp_method = CFComparisonMethod.from_str(method)

    if cmp_method is CFComparisonMethod.F:
        cmp = ExtraSumOfSquaresFTest(model_one, model_two, alpha)
    else:
        cmp = AICComparison(model_one, model_two, alpha)

    cmp.analyze()
    null = model_one.__class__.__name__
    alt = model_two.__class__.__name__
    cmp._report(null, alt)
    return cmp._result


class ModelComparison(Analysis):

    def __init__(
        self,
        model_one: CurveFit,
        model_two: CurveFit,
        alpha: float = 0.05,
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
        self._alpha = alpha

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
                    LOG.warning(
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
                    LOG.warning(
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

    @abstractmethod
    def _report(self, null_hyp: str, alt_hyp: str) -> None: ...


class ExtraSumOfSquaresFTest(ModelComparison):

    def __init__(
        self,
        model_one: CurveFit,
        model_two: CurveFit,
        alpha: float = 0.05,
    ) -> None:
        super().__init__(model_one, model_two, alpha)
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

    @override
    def _report(self, null_hyp: str, alt_hyp: str) -> None:
        table = PrettyTable()
        fields = [
            "Null hypothesis",
            "Alternative hypothesis",
            "P value",
            f"Conclusion (alpha = {self._alpha})",
            "Preferred model",
        ]
        table.add_column("", fields)

        for k, res in self._result.items():
            significant = res.p < self._alpha
            conclusion = "Reject" if significant else "Accept"
            preferred_model = alt_hyp if significant else null_hyp
            table.add_column(
                k,
                [
                    null_hyp,
                    alt_hyp,
                    f"{res.p:.2g}",
                    f"{conclusion} null hypothesis",
                    preferred_model,
                ],
            )

        table.align = "l"
        LOG.info(f"Comparison of fits\n{table}")


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

    @override
    def _report(self, null_hyp: str, alt_hyp: str) -> None:
        table = PrettyTable()
        fields = [
            "Model 1",
            "Model 1 probability",
            "Model 2",
            "Model 2 probability",
            "Ratio of probabilities",
            "Preferred model",
            "Difference in AICc",
        ]
        table.add_column("", fields)

        for k, res in self._result.items():
            preferred_model = alt_hyp if res.p2 > res.p1 else null_hyp
            table.add_column(
                k,
                [
                    null_hyp,
                    f"{res.p1:.2%}",
                    alt_hyp,
                    f"{res.p2:.2%}",
                    f"{res.p_ratio:.3g}",
                    preferred_model,
                    f"{res.delta_aicc:.3g}",
                ],
            )

        table.align = "l"
        LOG.info(f"Comparison of fits\n{table}")


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
