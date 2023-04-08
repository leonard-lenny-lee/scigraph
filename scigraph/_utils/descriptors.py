from typing import Any

import seaborn as sns

from ..graphing import cfg


class CfgProperty:
    """A property which has a default value in defaults.toml"""

    def __init__(self, name: str, cfg_key: str):
        self._public_name = name
        self._private_name = "_" + name
        self._cfg_key = cfg_key

    def __get__(self, __instance: Any, __owner: type | None = None) -> Any:
        # Use the cfg default if None is stored
        if not hasattr(__instance, self._private_name):
            return cfg[self._cfg_key]
        val = getattr(__instance, self._private_name)
        if val is None:
            return cfg[self._cfg_key]
        return val

    def __set__(self, __instance: Any, __value: Any) -> None:
        setattr(__instance, self._private_name, __value)

    def __delete__(self, __instance: Any) -> None:
        # Restore to default cfg value
        setattr(__instance, self._private_name, None)


class SnsPalette(CfgProperty):

    def __get__(self, __instance: Any, __owner: type | None = None) -> Any:
        return sns.color_palette(super().__get__(__instance, __owner))
