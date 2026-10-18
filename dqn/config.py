"""Config loading: YAML files with single inheritance plus command-line overrides.

A config file may contain `inherit: other.yaml` (path relative to the file).
Keys in the child override keys in the parent. Overrides from the command line
use `--set key=value`, where value is parsed as YAML (so 10000, 0.25, true work).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent is not None:
        base = load_config(path.parent / parent)
        base.update(cfg)
        cfg = base
    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item}")
        key, raw = item.split("=", 1)
        if key not in cfg:
            raise KeyError(f"Unknown config key '{key}'. Known keys: {sorted(cfg)}")
        cfg[key] = yaml.safe_load(raw)
    return cfg
