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

TEXT_SCHEMA = {
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

LINE_SCHEMA = {
    "length": Num,
    "width": Num,
    "color": String,
    "spacing_before": Num,
    "spacing_after": Num,
}

SCHEMA = {
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
