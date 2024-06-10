from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal, override

import numpy as np

from scigraph.analyses.abc import Analysis
from scigraph._options import (
    NormalizeSubColumnPolicy,
    NormalizeZeroPolicy,
    NormalizeOnePolicy,
    NormalizeResult,
)

if TYPE_CHECKING:
    from scigraph.datatables.abc import DataTable


class Normalize[T: DataTable](Analysis):

    def __init__(
        self,
        table: T,
        subcolumn_policy: Literal["average", "separate"] = "average",
        zero_policy: Literal["min", "first", "value"] = "min",
        one_policy: Literal["max", "last", "sum", "mean", "value"] = "max",
        zero_value: float = 0,
        one_value: float = 100,
        result: Literal["percentage", "fraction"] = "percentage",
        target_result: float = 1.0,
    ) -> None:
        self._table = table
        self._subcolumn_policy = NormalizeSubColumnPolicy.from_str(subcolumn_policy)
        self._zero_policy = NormalizeZeroPolicy.from_str(zero_policy)
        self._one_policy = NormalizeOnePolicy.from_str(one_policy)
        self._zero_value = zero_value
        self._one_value = one_value
        self._result = NormalizeResult.from_str(result)
        self._target_result = target_result

    @property
    @override
    def table(self) -> T:
        return self._table

    @override
    def analyze(self) -> T:
        out = deepcopy(self.table)

        if self._subcolumn_policy == NormalizeSubColumnPolicy.AVERAGE:
            values = np.vstack(
                [
                    np.nanmean(ds.y, axis=1)
                    for _, ds in self.table.datasets_itertuples()
                ]
            ).T
            n_repeats = round(
                self.table._get_normalize_values().shape[1] / values.shape[1]
            )
            values = values.repeat(n_repeats, axis=1)
        else:  # SEPARATE
            values = self.table._get_normalize_values()

        if self._zero_policy == NormalizeZeroPolicy.MIN:
            zero = np.nanmin(values, axis=0)
        elif self._zero_policy == NormalizeZeroPolicy.FIRST:
            zero = values[0]
        else:  # VALUE
            zero = np.full(values.shape[1], self._zero_value)

        if self._one_policy == NormalizeOnePolicy.MAX:
            one = np.nanmax(values, axis=0)
        elif self._one_policy == NormalizeOnePolicy.LAST:
            one = values[-1]
        elif self._one_policy == NormalizeOnePolicy.SUM:
            one = np.nansum(values, axis=0)
        elif self._one_policy == NormalizeOnePolicy.MEAN:
            one = np.nanmean(values, axis=0)
        else:  # VALUE
            one = np.full(values.shape[1], self._one_value)

        normalized = (self.table._get_normalize_values() - zero) / (one - zero)

        if self._result is NormalizeResult.PERCENTAGE:
            target = 100
        else:  # FRACTION
            target = self._target_result

        normalized *= target
        out._set_normalize_values(normalized)

        return out
