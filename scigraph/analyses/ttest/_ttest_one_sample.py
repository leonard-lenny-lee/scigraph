from __future__ import annotations

__all__ = ["OneSampleTTest"]

from typing import Optional, Literal, TYPE_CHECKING, override

import numpy as np
import pandas as pd
import scipy.stats

from scigraph.analyses.abc import GraphableAnalysis
from scigraph.analyses._utils import generate_p_summary
from scigraph.config import SG_DEFAULTS
from scigraph._options import TTestDirection

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from scigraph.datatables import ColumnTable
    from scigraph.graphs import ColumnGraph


class OneSampleTTest(GraphableAnalysis):

    def __init__(
        self,
        table: ColumnTable,
        hypothetical_value: float,
        direction: Literal["two sided", "greater", "less"] = "two sided",
        confidence_level: float = 0.95,
    ) -> None:
        self._table = table
        self._popmean = hypothetical_value
        self._direction = TTestDirection.from_str(direction)
        self._confidence_level = confidence_level

    @override
    def analyze(self) -> pd.DataFrame:
        res = scipy.stats.ttest_1samp(
            a=self.table.values, popmean=self._popmean, nan_policy="omit"
        )
        ds = self.table._dataset_names
        t = pd.Series(res.statistic, index=ds, name="t")  # type: ignore
        df = pd.Series(res.df, index=ds, name="df")  # type: ignore
        p = pd.Series(res.pvalue, index=ds, name=f"P value ({self._direction.to_str()})")  # type: ignore
        p_summary = pd.Series(
            [generate_p_summary(p_) for p_ in p], index=ds, name="P value summary"
        )
        alpha = 1 - self._confidence_level
        significant = pd.Series(
            res.pvalue < alpha,  # type: ignore
            index=ds,
            name=f"Significant ({alpha = :.2})?",
        )
        self._result = pd.concat([t, df, p, p_summary, significant], axis=1)
        return self._result

    @override
    def draw(
        self,
        graph: ColumnGraph,
        ax: Axes,
        space_above: Optional[float] = None,
        significant_only: bool = False,
        *_,
        **text_kws,
    ) -> None:
        if space_above is None:
            space_above = SG_DEFAULTS["analyses.ttest1samp.draw.space_above"]

        x = np.linspace(0, self.table.n_datasets, self.table.n_datasets, endpoint=False)
        y = self.table.values.max(axis=0)
        y = y + y.max() * space_above

        if graph._is_vertical:
            kws = {"va": "bottom", "ha": "center"}
        else:
            x, y = y, x
            kws = {"va": "center", "ha": "left", "rotation": -90}
        kws.update(text_kws)

        for x_, y_, summary in zip(x, y, self._result["P value summary"]):
            if significant_only and summary == "ns":
                continue
            ax.text(x_, y_, summary, None, **kws)

    @property
    @override
    def table(self) -> ColumnTable:
        return self._table
