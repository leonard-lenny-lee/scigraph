from __future__ import annotations

from typing import Any, NamedTuple


class Param[T](NamedTuple):

    ty: type[T] | tuple[type[T], ...]
    opt: set[T] | None

    def validate(self, arg: Any) -> bool:
        return (
            isinstance(arg, self.ty) 
            and (
                self.opt is None
                or arg in self.opt
            )
        )


# Generic Params
String = Param(str, None)
Int = Param(int, None)
Float = Param(float, None)
Num = Param((int, float), None)

type Schema = dict[str, Param | Schema]

TEXT_SCHEMA: Schema = {
    "size": Num,
    "weight": Param((int, float, str), None),
    "color": String,
    "spacing_before": Num,
    "spacing_after": Num,
    "linespacing": Num,
    "vertical_alignment": Param(
        str, {"top", "bottom", "baseline", "center", "center_baseline"}
    ),
    "horizontal_alignment": Param(str, {"left", "right", "center"}),
}

LINE_SCHEMA: Schema = {
    "length": Num,
    "width": Num,
    "color": String,
    "spacing_before": Num,
    "spacing_after": Num,
}

SCHEMA: Schema = {
    "datatables": {
        "xy": {
            "x_title": String,
            "y_title": String,
        },
        "column": {
            "x_title": String,
            "y_title": String,
        }
    },
    "graphs": {
        "axis": {
            "continuous": {
                "scale": Param(str, {"linear", "log10"}),
                "format": {
                    "linear": Param(str, {"decimal", "power10", "antilog"}),
                    "log10": Param(str, {"log10", "power10", "antilog"})
                }
            }
        }
    }
    ,
    "layout": {
        "caption": {
            "direction": Param(str, {"down", "up"}),
            "pad": {
                "top": Num,
                "left": Num,
                "right": Num,
                "bottom": Num,
            },
            "text": {
                "h1": TEXT_SCHEMA,
                "h2": TEXT_SCHEMA,
                "p": TEXT_SCHEMA,
                "s": TEXT_SCHEMA,
            },
            "line": {
                "dividing": LINE_SCHEMA,
                "branding": LINE_SCHEMA,
            }
        },
    },
}
