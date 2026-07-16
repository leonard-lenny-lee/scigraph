from enum import Enum, auto
from typing import Self


class Option(Enum):
    """Options which are parsed from user provided string literal arguments.
    All literal arguments in public APIs should be wrapped in an Option, to
    perform automatic checking. Use static analyzer for private interfaces.
    """

    def to_str(self, ws_char: str = "_") -> str:
        """Return the member name as a normalised lower-case string."""
        return self.name.replace("_", ws_char).lower()

    @classmethod
    def from_str(cls, s: str | Self) -> Self:
        """Parse a member from a case-insensitive, space-tolerant string."""
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
        """Return all available member names as normalised strings."""
        return set(cls.__members__.keys())


# fmt: off
## Graph Configuration

class ColumnGraphDirection(Option):
    """Orientation available for column and grouped graphs."""
    HORIZONTAL      = auto()
    VERTICAL        = auto()


GroupedGraphDirection = ColumnGraphDirection

class GroupedGraphGrouping(Option):
    """Arrangement available for grouped graph datasets."""
    INTERLEAVED     = auto()
    SEPARATED       = auto()
    STACKED         = auto()


## Graph Components ##

class PointsType(Option):
    """Summary or individual point representations."""
    MEAN            = auto()
    GEOMETRIC_MEAN  = auto()
    MEDIAN          = auto()
    INDIVIDUAL      = auto()
    SWARM           = auto()


class ErrorbarType(Option):
    """Supported error-bar summary statistics."""
    SD              = auto()
    GEOMETRIC_SD    = auto()
    SEM             = auto()
    CI95            = auto()
    RANGE           = auto()


class ConnectingLineType(Option):
    """Supported connecting-line summary statistics."""
    MEAN            = auto()
    GEOMETRIC_MEAN  = auto()
    MEDIAN          = auto()
    INDIVIDUAL      = auto()


class BarType(Option):
    """Supported bar summary statistics."""
    MEAN            = auto()
    GEOMETRIC_MEAN  = auto()
    MEDIAN          = auto()


LineType = BarType

## Analysis Configuration

class RowStatisticsScope(Option):
    """Scopes available to row-statistics analyses."""
    ROW             = auto()
    DATASET         = auto()


class DescStatsSubColPolicy(Option):
    """Policies for combining descriptive-statistics subcolumns."""
    AVERAGE         = auto()
    SEPARATE        = auto()
    MERGE           = auto()


class SummaryStatistic(Option):
    """Descriptive statistics available by name throughout the package."""
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
    """Alternative-hypothesis directions accepted by statistical tests."""
    TWO_SIDED       = auto()
    GREATER         = auto()
    LESS            = auto()


class WilcoxonZeroMethod(Option):
    """Zero-handling policies accepted by the Wilcoxon test."""
    WILCOX          = auto()
    PRATT           = auto()


class CFBoundType(Option):
    """Bound types available for curve-fit parameters."""
    EQUAL           = auto()
    GREATER         = auto()
    LESS            = auto()


class CFBandType(Option):
    """Uncertainty-band types available for curve fits."""
    CONFIDENCE      = auto()
    PREDICTION      = auto()


class CFReplicatePolicy(Option):
    """Policies for treating y replicates in curve fitting."""
    INDIVIDUAL      = auto()
    MEAN            = auto()


class CFConfidenceIntervalMethod(Option):
    """Methods available for parameter confidence-interval estimation."""
    PROFILE_LIKELIHOOD = auto()
    APPROXIMATE        = auto()
    BOOTSTRAP          = auto()
    NONE               = auto()


class CFComparisonMethod(Option):
    """Model-comparison methods available for curve fits."""
    AIC             = auto()
    F               = auto()


class CFCompareDiff(Option):
    """Ways to express between-dataset parameter differences."""
    DIFF            = auto()
    ABS             = auto()
    FOLD            = auto()


class NormalizeSubColumnPolicy(Option):
    """Policies for combining replicate subcolumns during normalisation."""
    AVERAGE         = auto()
    SEPARATE        = auto()


class NormalizeZeroPolicy(Option):
    """Reference-value policies for the zero point of normalisation."""
    MIN             = auto()
    FIRST           = auto()
    VALUE           = auto()


class NormalizeOnePolicy(Option):
    """Reference-value policies for the one point of normalisation."""
    MAX             = auto()
    LAST            = auto()
    SUM             = auto()
    MEAN            = auto()
    VALUE           = auto()


class NormalizeResult(Option):
    """Output scales available for normalisation."""
    PERCENTAGE      = auto()
    FRACTION        = auto()

# fmt: on
