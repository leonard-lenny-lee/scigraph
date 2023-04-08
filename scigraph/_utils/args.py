
from enum import Enum


class Arg(Enum):

    def __str__(self) -> str:
        return self.name.lower()

    @classmethod
    def from_str(cls, s: str):
        r = cls.__members__.get(s.upper(), None)
        if r is None:
            valid_args = ", ".join(cls.__members__.keys())
            arg_name = cls.__name__.lower().replace("_", "")
            raise ValueError(
                f"Invalid {arg_name} argument '{s}'. "
                f"Valid arguments are '{valid_args}'"
            )
        return r
