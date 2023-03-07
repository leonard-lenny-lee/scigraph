
import json
import os
import toml
from types import MappingProxyType
from typing import Any, Dict, Set
from warnings import warn

import seaborn as sns

CONFIG_FILE = "defConfig.toml"


class Config:

    _instance = None

    def __init__(self):
        raise RuntimeError("Call instance()")

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance._load_default()
        return cls._instance

    def __getitem__(self, key: str) -> Any:
        keys = key.split(".")
        node = self._config
        for k in keys:
            node = node[k]
        if isinstance(node, dict):
            return MappingProxyType(node)
        return node

    def __setitem__(self, key: str, val: Any) -> None:
        keys = key.split(".")
        n_keys = len(keys) - 1
        node = self._config
        for i, k in enumerate(keys):
            if i < n_keys:
                node = node[k]
            else:
                node[k] = val

    def __str__(self) -> str:
        return json.dumps(self._config, sort_keys=True, indent=4)

    def load(self, f: str) -> None:
        custom_cfg = self._flatten(toml.load(f))
        invalid_keys = []
        for k, v in custom_cfg.items():
            if k in self._keys:
                self.__setitem__(k, v)
            else:
                invalid_keys.append(k)
        if invalid_keys:
            display_keys = "\n".join(invalid_keys)
            warn(
                f"Unrecognised keys in config file ignored:\n{display_keys}",
                RuntimeWarning
            )

    @property
    def keys(self) -> Set[str]:
        return self._keys

    def _load_default(self) -> None:
        dir = os.path.dirname(__file__)
        filename = os.path.join(dir, CONFIG_FILE)
        self._config = toml.load(filename)
        self._keys = set(self._flatten(self._config).keys())

    def _flatten(
        self,
        node: Dict[str, Any],
        keypath: str = None,
        cfgs: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if cfgs is None:
            cfgs = {}
        for k, v in node.items():
            key = k if keypath is None else f"{keypath}.{k}"
            if isinstance(v, dict):
                self._flatten(v, key, cfgs)
            else:
                cfgs[key] = node[k]
        return cfgs


cfg = Config.instance()
# Set default styling scheme
sns.set(style="ticks", palette="Set2")
sns.despine()
