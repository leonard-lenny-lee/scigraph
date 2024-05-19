from __future__ import annotations

from typing import Any, NamedTuple


class Param[T](NamedTuple):

    ty: type[T] | tuple[type[T], ...]
    variadic: bool = False
    opt: set[T] | None = None

    def validate(self, arg: Any) -> bool:
        return isinstance(arg, self.ty) and (self.opt is None or arg in self.opt)


# Generic Params
String = Param(str)
Int = Param(int)
Float = Param(float)
Num = Param((int, float))

VString = Param(str, variadic=True)
VNum = Param((int, float), variadic=True)

type Schema = dict[str, Param | Schema]


TEXT_SCHEMA: Schema = {
    "size": Num,
    "weight": Param((int, float, str)),
    "color": String,
    "spacing_before": Num,
    "spacing_after": Num,
    "linespacing": Num,
    "vertical_alignment": Param(
        str, opt={"top", "bottom", "baseline", "center", "center_baseline"}
    ),
    "horizontal_alignment": Param(str, opt={"left", "right", "center"}),
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
        },
        "grouped": {
            "x_title": String,
            "y_title": String,
        },
    },
    "graphs": {
        "xy": {
            "color": VString,
            "ls": VString,
            "marker": VString,
            "linewidth": VNum,
            "markersize": VNum,
            "capsize": VNum,
            "capthickness": VNum,
        },
        "column": {
            "color": VString,
            "ls": VString,
            "marker": VString,
            "linewidth": VNum,
            "markersize": VNum,
            "capsize": VNum,
            "capthickness": VNum,
            "barcolor": VString,
            "barwidth": VNum,
            "baredgecolor": VString,
            "baredgethickness": VNum,
        },
        "grouped": {
            "color": VString,
            "ls": VString,
            "marker": VString,
            "linewidth": VNum,
            "markersize": VNum,
            "capsize": VNum,
            "capthickness": VNum,
            "barcolor": VString,
            "barwidth": VNum,
            "baredgecolor": VString,
            "baredgethickness": VNum,
        },
        "axis": {
            "continuous": {
                "scale": Param(str, opt={"linear", "log10"}),
                "format": {
                    "linear": Param(str, opt={"decimal", "power10", "antilog"}),
                    "log10": Param(str, opt={"log10", "power10", "antilog"}),
                },
            }
        },
    },
    "analyses": {
        "ttest": {
            "draw": {
                "arm_length": Num,
                "distance_below": Num,
                "distance_above": Num,
                "color": String,
                "linewidth": Num,
            }
        },
        "curve_fit": {
            "bands": {
                "alpha": Num,
                "linewidth": Num,
                "linestyle": String,
            }
        },
    },
    "layout": {
        "caption": {
            "direction": Param(str, opt={"down", "up"}),
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
            },
        },
    },
}
