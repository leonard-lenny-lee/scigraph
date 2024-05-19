from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import os
import json
import tomllib
from typing import Any, Self

from ._schema import SCHEMA, Param
from scigraph._log import LOG


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

    def query_schema(self, key: str | None) -> Any:
        if key is None:
            return deepcopy(SCHEMA)
        keys = key.split(".")
        out = SCHEMA

        for k in keys:
            if not isinstance(out, dict) or k not in out:
                raise KeyError(f"Invalid key: {key}")
            out = out[k]

        # Return deepcopy to prevent unchecked mutation of internal schema
        return deepcopy(out)

    def set(self, key: str, val: Any) -> None:
        # Validate key and value before attempting to set value
        query_result = self.query_schema(key)
        if not isinstance(query_result, Param):
            raise KeyError(f"Invalid key: {key}. Must be a parameter.")
        if not query_result.validate(val):
            raise KeyError(f"Invalid value: {val}")

        # Set value
        keys = key.split(".")
        target = self._config
        for k in keys[:-1]:
            target = target[k]
        assert not isinstance(target[keys[-1]], dict)
        target[keys[-1]] = val

    def load_toml(self, fp: str | Path) -> None:
        fp = self._find_file(fp)
        if fp.suffix != ".toml":
            raise ValueError("Must be a TOML file")

        with open(fp, "rb") as f:
            d = tomllib.load(f)

        errors: list[ConfigParseError] = []
        self._validate(d, SCHEMA, errors, "")

        error_count = 0
        for error in errors:
            match error:
                case MissingParam(_):
                    continue
                case (
                    UnknownParam(k, _)
                    | InvalidParamType(k, _, _)
                    | InvalidParamValue(k, _, _)
                ):
                    LOG.warn(error)
                    # Remove invalid configuration settings
                    error_count += 1
                    keys = k.split(".")
                    d_ = d
                    for k in keys[:-1]:
                        d_ = d_[k]
                    del d_[keys[-1]]

        if error_count:
            LOG.warn(f"{error_count} configuration options have been ignored.")

        self._flatten(d, (d_flat := {}), "")

        for kv in d_flat.items():
            self.set(*kv)

    def reset(self) -> None:
        self._config = self._load_default()._config

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
            LOG.error(str(error))
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
                self._validate(d_val, schema_val, errors, prefix + key)
                continue

            assert isinstance(schema_val, Param)

            # Terminal node
            # Put values in a list to handle variadic configuration fields
            if schema_val.variadic and isinstance(d_val, list):
                to_check = d_val
            else:
                to_check = [d_val]

            for item in to_check:
                # Ensure type is correct
                if not isinstance(item, schema_val.ty):
                    errors.append(
                        InvalidParamType(
                            param_name=key_path,
                            expected_type=schema_val.ty,
                            provided_type=type(item),
                        )
                    )
                    continue
                # Ensure value is valid
                if schema_val.opt is not None and item not in schema_val.opt:
                    errors.append(
                        InvalidParamValue(
                            param_name=key_path,
                            provided_value=item,
                            valid_values=schema_val.opt,
                        )
                    )
                # Passed all validation checks

        # Check for unknown parameters
        unknown_keys = d.keys() - schema_d.keys()
        for unknown_key in unknown_keys:
            errors.append(
                UnknownParam(
                    param_name=prefix + unknown_key,
                    value=d[unknown_key],
                )
            )

    def _find_file(self, fp: str | Path) -> Path:
        # Look in current directory
        p = Path(__file__).resolve().parent.joinpath(fp)
        if os.path.isfile(p):
            return p
        # Interpret as absolute
        p = Path(fp).absolute()
        if os.path.isfile(p):
            return p
        raise FileExistsError(fp)

    def _flatten(self, d: dict[str, Any], out: dict[str, Any], prefix: str) -> None:
        if prefix != "":
            prefix += "."

        for k, v in d.items():
            key = prefix + k
            if isinstance(v, dict):
                self._flatten(v, out, key)
                continue
            out[key] = v

    @classmethod
    def _load_default(cls) -> Self:
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

SG_DEFAULTS = DefaultsConfiguration._load_default()
