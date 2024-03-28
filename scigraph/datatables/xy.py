from enum import Enum
import math

from numpy.typing import NDArray
from pandas import DataFrame, MultiIndex, concat

from .abc import DataTable

DEFAULT_XNAME = "X"
DEFAULT_YNAME = "Y"


class XOpt(Enum):
    NUMS = 1
    NUMS_AND_ERRORS = 2


class YOpt(Enum):
    SINGLE = 1
    REPLICATES = 2


class XYTable(DataTable):

    def __init__(
        self,
        values: NDArray,
        xopt: XOpt = XOpt.NUMS,
        yopt: YOpt = YOpt.REPLICATES,
        n_replicates: int = 4,  # Ignored if y_opt != "replicate"
    ) -> None:
        if values.ndim != 2:
            raise ValueError("Expected 2D Matrix")
        self._values = values
        self._xopt = xopt
        self._yopt = yopt
        self._n_replicates = n_replicates if yopt == YOpt.REPLICATES else 1
        self.xname = DEFAULT_XNAME
        self.yname = DEFAULT_YNAME
        self.ygroups = self._default_names(n=self.ngroups)

    def as_df(self) -> DataFrame:
        ycols = MultiIndex.from_product(
            [self.ygroups, range(1, self._n_replicates + 1)]
        )
        y = DataFrame(self.yvalues, columns=ycols)
        xsubcols = ["X"]
        if self._xopt == XOpt.NUMS_AND_ERRORS:
            xsubcols.append("Err. Bars")
        assert len(xsubcols) == self.n_xcols
        xcols = MultiIndex.from_product([[self.xname], xsubcols])
        x = DataFrame(self.xvalues, columns=xcols)
        return concat([x, y], axis=1)

    @property
    def values(self) -> NDArray:
        return self._values

    @property
    def xopt(self) -> XOpt:
        return self._xopt

    @property
    def yopt(self) -> YOpt:
        return self._yopt

    @property
    def n_replicates(self) -> int:
        return self._n_replicates

    @property
    def nrows(self) -> int:
        return self._values.shape[0]

    @property
    def ncols(self) -> int:
        return self._values.shape[1]

    @property
    def n_xcols(self) -> int:
        match self._xopt:
            case XOpt.NUMS:
                n = 1
            case XOpt.NUMS_AND_ERRORS:
                n = 2
            case _:
                raise NotImplementedError
        return n

    @property
    def n_ycols(self) -> int:
        return self.ncols - self.n_xcols

    @property
    def ngroups(self) -> int:
        match self._yopt:
            case YOpt.SINGLE:
                n = self.n_ycols
            case YOpt.REPLICATES:
                n = math.ceil(self.n_ycols / self._n_replicates)
            case _:
                raise NotImplementedError
        return n

    @property
    def ygroups(self) -> list[str]:
        return self._ygroups

    @ygroups.setter
    def ygroups(self, val: list[str]) -> None:
        if len(val) != self.ngroups:
            raise ValueError
        self._ygroups = val

    @property
    def yvalues(self) -> NDArray:
        return self._values[:, self.n_xcols:]

    @property
    def xvalues(self) -> NDArray:
        return self._values[:, :self.n_xcols]

    @property
    def xerr(self) -> NDArray | None:
        if self.xopt == XOpt.NUMS:
            return None
        return self.xvalues[:, -1]
