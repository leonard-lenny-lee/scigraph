from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional, TYPE_CHECKING, override

import matplotlib.lines as mlines
import scipy.stats

from scigraph.analyses.abc import GraphableAnalysis
from scigraph.analyses._utils import generate_p_summary
from scigraph._options import TTestDirection
from scigraph.config import SG_DEFAULTS

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from scigraph.datatables import ColumnTable
    from scigraph.graphs import ColumnGraph


class TTest(GraphableAnalysis):
    """Base class for t-tests and other non-parametric tests"""

    def __init__(
        self,
        table: ColumnTable,
        datasets: tuple[str, str],
        direction: Literal["two sided", "greater", "less"] = "two sided",
        confidence_level: float = 0.95,
    ) -> None:
        self._table = table
        self._datasets = datasets
        self._direction = TTestDirection.from_str(direction)
        self._confidence_level = confidence_level

    @override
    def draw(
        self,
        graph: ColumnGraph,
        ax: Axes,
        arm_length: Optional[float] = None,
        distance_below: Optional[float] = None,
        distance_above: Optional[float] = None,
        color: Optional[str] = None,
        linewidth: Optional[float] = None,
        *args,
        **kwargs,
    ) -> None:
        # Apply defaults
        defaults = SG_DEFAULTS["analyses.ttest.draw"]
        if arm_length is None:
            arm_length = defaults["arm_length"]
        if distance_below is None:
            distance_below = defaults["distance_below"]
        if distance_above is None:
            distance_above = defaults["distance_above"]
        if color is None:
            color = defaults["color"]
        if linewidth is None:
            linewidth = defaults["linewidth"]
        line_kws = {"color": color, "linewidth": linewidth}

        x0, x1 = [graph.table.dataset_ids.index(d) for d in self._datasets]
        x2 = (x0 + x1) / 2
        a, b = self._get_ab()
        t_max = graph.table.values.max()
        y_max = max(a.max(), b.max())
        y0 = y_max + distance_below * t_max
        y1 = y0 + arm_length * t_max
        y2 = y1 + distance_above * t_max

        x_a, y_a = [x0, x1], [y1, y1]
        x_b, y_b = [x0, x0], [y0, y1]
        x_c, y_c = [x1, x1], [y0, y1]

        if not graph._is_vertical:
            x_a, x_b, x_c, y_a, y_b, y_c = y_a, y_b, y_c, x_a, x_b, x_c
            x2, y2 = y2, x2
            text_kws = {"va": "center", "ha": "left", "rotation": -90}
        else:
            text_kws = {"va": "bottom", "ha": "center"}

        # Main line
        ax.add_line(mlines.Line2D(x_a, y_a, **line_kws))
        # Arms
        ax.add_line(mlines.Line2D(x_b, y_b, **line_kws))
        ax.add_line(mlines.Line2D(x_c, y_c, **line_kws))
        # Label
        ax.text(x2, y2, self.result._label, None, **text_kws)

    @property
    @override
    def table(self) -> ColumnTable:
        return self._table

    def _get_ab(self) -> tuple[NDArray, NDArray]:
        a = self.table.get_dataset(self._datasets[0]).y
        b = self.table.get_dataset(self._datasets[1]).y
        return a, b

    def _ttest(self, ttest, **kwargs) -> TTestResult:
        a, b = self._get_ab()
        alternative = self._direction.to_str('-')
        r = ttest(a, b, nan_policy='omit', alternative=alternative, **kwargs)
        t, p, df = r.statistic, r.pvalue, r.df  # type: ignore
        t: float; p: float; df: float

        p_summary = generate_p_summary(p)
        signficant = p < self._confidence_level
        a_mean, b_mean = a.mean(), b.mean()
        mean_difference = a_mean - b_mean
        ci = r.confidence_interval()

        return TTestResult(p, p_summary, signficant, self._confidence_level,
                           alternative, t, df, a_mean, b_mean, mean_difference,
                           ci)

    @property
    @override
    def result(self) -> _Result:
        return super().result

class _Result(ABC):
    
    @property
    @abstractmethod
    def _label(self) -> str: ...


@dataclass(frozen=True)
class TTestResult( _Result):
    p: float
    p_summary: str
    significant: bool
    confidence_level: float
    direction: str
    t: float
    df: float
    a_mean: float
    b_mean: float
    mean_difference: float
    confidence_interval: tuple[float, float]

    @property
    @override
    def _label(self) -> str:
        return self.p_summary


class StudentsTTest(TTest):

    @override
    def analyze(self) -> TTestResult:
        self._result = self._ttest(scipy.stats.ttest_ind, equal_var=True)
        return self._result


class WelchTTest(TTest):

    @override
    def analyze(self) -> TTestResult:
        self._result = self._ttest(scipy.stats.ttest_ind, equal_var=False)
        return self._result


class PairedTTest(TTest):

    @override
    def analyze(self) -> TTestResult:
        self._result = self._ttest(scipy.stats.ttest_rel)
        return self._result


class MannWhitneyTest(TTest): ...


class KolmogorovSmirnovTest(TTest): ...
