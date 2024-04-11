from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
import tomllib
from typing import Any, Self
import logging

from _schema import SCHEMA, Param


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.joinpath("default.toml")


class DefaultsConfiguration:

    def __init__(self, d: dict) -> None:
        self._strict_validate(d, SCHEMA)
        self._config = d

    def __str__(self) -> str:
        return json.dumps(self._config, indent=4)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, val: Any) -> None:
        return self.set(key, val)

    def get(self, key: str | None) -> Any:
        if key is None:
            return deepcopy(self._config)
        keys = key.split(".")
        out = self._config

        for k in keys:
            if not isinstance(out, dict) or k not in out:
                raise KeyError(f"Invalid key: {key}")
            out = out[k]

        # Return deepcopy to prevent unchecked mutation of internal configuration
        return deepcopy(out)

    def set(self, key: str, val: Any) -> None:
        keys = key.split(".")

        # Validate key and value before attempting to set value
        cur = SCHEMA
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                raise KeyError(f"Invalid key: {key}")
            cur = cur[k]
        if not isinstance(cur, Param):
            raise KeyError(f"Invalid key: {key}")
        if not cur.validate(val):
            raise KeyError(f"Invalid value: {val}")

        # Set value
        target = self._config
        for k in keys[:-1]:
            target = target[k]
        assert not isinstance(target[keys[-1]], dict)
        target[keys[-1]] = val

    def _strict_validate(
        self,
        d: dict[str, Any],
        schema_d: dict[str, Any],
    ) -> None:
        errors = []
        self._validate(d, schema_d, errors, "")
        valid = len(errors) == 0
        if valid:
            return
        for error in errors:
            logging.error(str(error))
        raise RuntimeError("Invalid Configuration File")

    def _validate(
        self,
        d: dict[str, Any],
        schema_d: dict[str, Any],
        errors: list[ConfigParseError],
        prefix: str,
    ) -> None:
        if prefix != "":
            prefix += "."

        for key, schema_val in schema_d.items():
            key_path = prefix + key
            # Ensure correct key is provided
            if key not in d:
                errors.append(MissingParam(key_path))
                continue
            
            d_val = d[key]
            if isinstance(schema_val, dict):
                # Branch node
                if not isinstance(d_val, dict):
                    errors.append(
                        InvalidParamType(
                            param_name=key_path,
                            expected_type=dict,
                            provided_type=type(d_val),
                        )
                    )
                    continue
                self._validate(d_val, schema_val, errors, prefix+key)
                continue

            assert isinstance(schema_val, Param)
            # Terminal node
            # Ensure type is correct
            if not isinstance(d_val, schema_val.ty):
                errors.append(
                    InvalidParamType(
                        param_name=key_path,
                        expected_type=schema_val.ty,
                        provided_type=type(d_val)
                    )
                )
                continue
            # Ensure value is valid
            if schema_val.opt is not None and d_val not in schema_val.opt:
                errors.append(
                    InvalidParamValue(
                        param_name=key_path,
                        provided_value=d_val,
                        valid_values=schema_val.opt,
                    )
                )
            # Passed all validation checks

        # Check for unknown parameters
        unknown_keys = d.keys() - schema_d.keys()
        for unknown_key in unknown_keys:
            errors.append(
                UnknownParam(
                    param_name=prefix+unknown_key,
                    value=d[unknown_key],
                )
            )

    @classmethod
    def load_default(cls) -> Self:
        return cls._from_toml(DEFAULT_CONFIG_PATH)

    @classmethod
    def _from_toml(cls, fp: str | Path) -> Self:
        with open(fp, "rb") as f:
            d = tomllib.load(f)
        return cls(d)


@dataclass
class MissingParam:
    param_name: str


@dataclass
class UnknownParam:
    param_name: str
    value: Any


@dataclass
class InvalidParamType:
    param_name: str
    expected_type: type | tuple[type]
    provided_type: type


@dataclass
class InvalidParamValue:
    param_name: str
    provided_value: Any
    valid_values: set[Any]


type ConfigParseError = MissingParam | UnknownParam | InvalidParamType | InvalidParamValue

SG_DEFAULTS = DefaultsConfiguration.load_default()

if __name__ == "__main__":
    # PARSE TESTING
    cfg = DefaultsConfiguration._from_toml(DEFAULT_CONFIG_PATH)
    print(cfg)
