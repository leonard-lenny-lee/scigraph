from enum import Enum
from typing import Self


class GraphType(Enum):

    XY = 0
    COLUMN = 1


class Option(Enum):
    """Options which are parsed from user provided string literal arguments."""

    def to_str(self) -> str:
        return self.name.replace("_", " ").lower()

    @classmethod
    def from_str(cls, s: str | Self) -> Self:
        if isinstance(s, cls):
            return s
        if not isinstance(s, str):
            raise TypeError(
                f"{cls.__name__.lower()} argument must be a string."
            )
        key = s.upper().strip().replace(' ', '_')
        if key in cls.__members__:
            return cls[key]
        else:
            opts = [m.lower() for m in cls.__members__.keys()]
            raise ValueError(
                f"Invalid {cls.__name__.lower()} argument: '{s}'. "
                f"Options: {opts}"
            )


class XYGraphSubtype(Option):

    MEAN = 0
    GEOMETRIC_MEAN = 1
    MEDIAN = 2
    INDIVIDUAL = 3


class ColumnGraphDirection(Option):
    
    HORIZONTAL = 0
    VERTICAL = 1


class ColumnGraphSubtype(Option):

    MEAN = 0
    GEOMETRIC_MEAN = 1
    MEDIAN = 2
    INDIVIDUAL = 3
    SWARM = 4


class BarType(Option):

    MEAN = 0
    GEOMETRIC_MEAN = 1
    MEDIAN = 2


LineType = BarType
