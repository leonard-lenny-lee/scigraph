from enum import Enum
from typing import Self


class PtOpt(Enum):
    MEAN = 0
    MEDIAN = 1
    INDIVIDUAL = 2


class ErrOpt(Enum):
    SD = 0
    SEM = 1
    CI95 = 2
    RANGE = 3
    IQR = 4
    NONE = 5


class ScaleOpt(Enum):
    LINEAR = 0
    LOG10 = 1

    def mpl_arg(self) -> str:
        match self:
            case self.LINEAR:
                out = "linear"
            case self.LOG10:
                out = "log"
            case _:
                raise NotImplementedError
        return out

    @classmethod
    def default(cls) -> Self:
        return cls.LINEAR


class Log10ScaleFmtOpt(Enum):
    ANTILOG = 0
    POWER10 = 1
    LOG = 2

    @classmethod
    def default(cls) -> Self:
        return cls.POWER10
