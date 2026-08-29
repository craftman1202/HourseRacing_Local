"""conf/*.yaml のロードと、設定間の導出関係の強制。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONF_DIR = Path(__file__).resolve().parents[2] / "conf"

_ENV_RE = re.compile(r"\$\{env:([A-Z_][A-Z0-9_]*)(?:,\s*([^}]*))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def sub(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), (m.group(2) or "").strip())
        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_yaml(name: str, conf_dir: Path | None = None) -> dict[str, Any]:
    path = (conf_dir or CONF_DIR) / f"{name}.yaml"
    return _expand(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class CVConfig:
    """walk-forward 分割の設定。

    embargo_days は features.yaml の max_lookback_days から導出される。cv.yaml に
    embargo を直接書く経路は存在しない — 二重管理は必ずズレるため（CV-04）。
    """

    mode: str
    train_start: date
    folds: list[tuple[date, date]]
    oos: tuple[date, date | None]
    oos_locked: bool
    inner_n_folds: int
    rolling_window_years: int
    embargo_days: int


@dataclass(frozen=True)
class FeatureConfig:
    max_lookback_days: int
    track: str
    train_from: date
    shrinkage: dict[str, float]
    windows: dict[str, int]
    market_term_blocklist: list[str]
    exclude_baba_codes: list[int] = field(default_factory=list)


def _d(v: str) -> date:
    return date.fromisoformat(v)


@lru_cache(maxsize=None)
def feature_config(conf_dir: str | None = None) -> FeatureConfig:
    raw = load_yaml("features", Path(conf_dir) if conf_dir else None)
    windows = raw["windows"]
    if windows["rolling_days"] > raw["max_lookback_days"]:
        raise ValueError(
            f"rolling_days={windows['rolling_days']} が max_lookback_days="
            f"{raw['max_lookback_days']} を超えています。embargo が実際の"
            "ルックバックより短くなりリークします。"
        )
    return FeatureConfig(
        max_lookback_days=int(raw["max_lookback_days"]),
        track=raw["track"],
        train_from=_d(raw["train_from"]),
        shrinkage=raw["shrinkage"],
        windows=windows,
        market_term_blocklist=raw["market_term_blocklist"],
        exclude_baba_codes=raw.get("exclude", {}).get("baba_codes", []),
    )


@lru_cache(maxsize=None)
def cv_config(conf_dir: str | None = None) -> CVConfig:
    raw = load_yaml("cv", Path(conf_dir) if conf_dir else None)
    if "embargo_days" in raw:
        raise ValueError(
            "cv.yaml に embargo_days を書かないでください。features.yaml の "
            "max_lookback_days から自動導出されます（CV-04）。"
        )
    feats = feature_config(conf_dir)
    oos = raw["oos"]
    return CVConfig(
        mode=raw["mode"],
        train_start=_d(raw["train_start"]),
        folds=[(_d(f["valid_start"]), _d(f["valid_end"])) for f in raw["folds"]],
        oos=(_d(oos["valid_start"]), _d(oos["valid_end"]) if oos.get("valid_end") else None),
        oos_locked=bool(oos.get("locked", True)),
        inner_n_folds=int(raw["inner"]["n_folds"]),
        rolling_window_years=int(raw["rolling_window_years"]),
        embargo_days=feats.max_lookback_days,
    )


@lru_cache(maxsize=None)
def data_config(conf_dir: str | None = None) -> dict[str, Any]:
    return load_yaml("data", Path(conf_dir) if conf_dir else None)


@lru_cache(maxsize=None)
def eda_config(conf_dir: str | None = None) -> dict[str, Any]:
    return load_yaml("eda", Path(conf_dir) if conf_dir else None)


def reset_cache() -> None:
    """テストで conf を差し替えるとき用。"""
    for fn in (feature_config, cv_config, data_config, eda_config):
        fn.cache_clear()
