from enum import Enum, auto
from typing import Self


class Option(Enum):
    """Options which are parsed from user provided string literal arguments.
    All literal arguments in public APIs should be wrapped in an Option, to
    perform automatic checking. Use static analyzer for private interfaces.
    """

    def to_str(self, ws_char: str = "_") -> str:
        return self.name.replace("_", ws_char).lower()

    @classmethod
    def from_str(cls, s: str | Self) -> Self:
        if isinstance(s, cls):
            return s
        if not isinstance(s, str):
            raise TypeError(f"{cls.__name__.lower()} argument must be a string.")
        key = s.upper().strip().replace(" ", "_")
        if key in cls.__members__:
            return cls[key]
        else:
            opts = [m.lower() for m in cls.__members__.keys()]
            raise ValueError(
                f"Invalid {cls.__name__.lower()} argument: '{s}'. " f"Options: {opts}"
            )

    @classmethod
    def to_strs(cls) -> set[str]:
        return set(cls.__members__.keys())


# fmt: off
## Graph Configuration

class ColumnGraphDirection(Option):
    HORIZONTAL      = auto()
    VERTICAL        = auto()


GroupedGraphDirection = ColumnGraphDirection

class GroupedGraphGrouping(Option):
    INTERLEAVED     = auto()
    SEPARATED       = auto()
    STACKED         = auto()


## Graph Components ##

class PointsType(Option):
    MEAN            = auto()
    GEOMETRIC_MEAN  = auto()
    MEDIAN          = auto()
    INDIVIDUAL      = auto()
    SWARM           = auto()


class ErrorbarType(Option):
    SD              = auto()
    GEOMETRIC_SD    = auto()
    SEM             = auto()
    CI95            = auto()
    RANGE           = auto()


class ConnectingLineType(Option):
    MEAN            = auto()
    GEOMETRIC_MEAN  = auto()
    MEDIAN          = auto()
    INDIVIDUAL      = auto()


class BarType(Option):
    MEAN            = auto()
    GEOMETRIC_MEAN  = auto()
    MEDIAN          = auto()


LineType = BarType

## Analysis Configuration

class RowStatisticsScope(Option):
    ROW             = auto()
    DATASET         = auto()


class DescStatsSubColPolicy(Option):
    AVERAGE         = auto()
    SEPARATE        = auto()
    MERGE           = auto()


class SummaryStatistic(Option):
    MEAN            = auto()
    SD              = auto()
    SEM             = auto()
    SUM             = auto()
    MIN             = auto()
    MAX             = auto()
    RANGE           = auto()
    N               = auto()
    LOWER_QUARTILE  = auto()
    UPPER_QUARTILE  = auto()
    MEDIAN          = auto()
    CV              = auto()
    SKEWNESS        = auto()
    KURTOSIS        = auto()
    GEOMETRIC_MEAN  = auto()
    GEOMETRIC_SD    = auto()
    MEAN_CI         = auto()


class TTestDirection(Option):
    TWO_SIDED       = auto()
    GREATER         = auto()
    LESS            = auto()


class WilcoxonZeroMethod(Option):
    WILCOX          = auto()
    PRATT           = auto()


class CFBoundType(Option):
    EQUAL           = auto()
    GREATER         = auto()
    LESS            = auto()


class CFBandType(Option):
    CONFIDENCE      = auto()
    PREDICTION      = auto()


class CFReplicatePolicy(Option):
    INDIVIDUAL      = auto()
    MEAN            = auto()


class CFComparisonMethod(Option):
    AIC             = auto()
    F               = auto()


class CFCompareDiff(Option):
    DIFF            = auto()
    ABS             = auto()
    FOLD            = auto()


class NormalizeSubColumnPolicy(Option):
    AVERAGE         = auto()
    SEPARATE        = auto()


class NormalizeZeroPolicy(Option):
    MIN             = auto()
    FIRST           = auto()
    VALUE           = auto()


class NormalizeOnePolicy(Option):
    MAX             = auto()
    LAST            = auto()
    SUM             = auto()
    MEAN            = auto()
    VALUE           = auto()


class NormalizeResult(Option):
    PERCENTAGE      = auto()
    FRACTION        = auto()

# fmt: on
