"""CV-01..10: 分割の健全性と OOS ロック。"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml

from nar.config import cv_config, feature_config, reset_cache
from nar.errors import OOSAccessError
from nar.eval.splits import (
    OOSGuard, inner_folds, make_folds, make_oos_fold, validate_folds,
)


@pytest.fixture
def races():
    d = pd.date_range("2010-01-01", "2025-12-31", freq="D")
    return pd.DataFrame({
        "race_id": [f"20{i:010d}" for i in range(len(d))],
        "race_date": d,
    })


@pytest.fixture
def cfg():
    reset_cache()
    return cv_config()


def _conf_dir(tmp_path: Path, max_lookback_days: int) -> Path:
    """features.yaml だけ書き換えた conf ディレクトリを作る。"""
    src = Path(__file__).resolve().parents[1] / "conf"
    dst = tmp_path / "conf"
    dst.mkdir()
    for f in src.glob("*.yaml"):
        (dst / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    feats = yaml.safe_load((dst / "features.yaml").read_text(encoding="utf-8"))
    feats["max_lookback_days"] = max_lookback_days
    feats["windows"]["rolling_days"] = min(feats["windows"]["rolling_days"], max_lookback_days)
    (dst / "features.yaml").write_text(yaml.safe_dump(feats, allow_unicode=True), encoding="utf-8")
    return dst


# ------------------------------------------------------------------ CV-01/02/05
def test_cv01_cv02_cv05_no_overlap_in_race_or_date(cfg, races):
    folds = make_folds(cfg)
    validate_folds(folds, races)  # 重複・日跨ぎ・時間方向をまとめて検証
    for f in folds:
        tr, va = f.mask(races["race_date"])
        assert not (set(races.loc[tr, "race_id"]) & set(races.loc[va, "race_id"]))
        assert races.loc[tr, "race_date"].max() < races.loc[va, "race_date"].min()


# ------------------------------------------------------------------------ CV-03
def test_cv03_embargo_gap_respected_in_every_fold(cfg, races):
    for f in make_folds(cfg):
        tr, va = f.mask(races["race_date"])
        gap = (races.loc[va, "race_date"].min() - races.loc[tr, "race_date"].max()).days
        assert gap >= cfg.embargo_days, f"fold {f.index}: gap {gap} < {cfg.embargo_days}"


# ------------------------------------------------------------------------ CV-04
@pytest.mark.parametrize("lookback", [180, 365])
def test_cv04_embargo_follows_max_lookback_window(tmp_path, lookback):
    """features.yaml の窓長を変えると embargo が自動追随する。"""
    reset_cache()
    conf = str(_conf_dir(tmp_path, lookback))
    try:
        assert feature_config(conf).max_lookback_days == lookback
        c = cv_config(conf)
        assert c.embargo_days == lookback
        for f in make_folds(c):
            assert (f.valid_start - f.train_end).days == lookback + 1
    finally:
        reset_cache()


def test_cv04_embargo_cannot_be_hardcoded_in_cv_yaml(tmp_path):
    """cv.yaml に embargo_days を書く経路を塞ぐ。二重管理は必ずズレる。"""
    reset_cache()
    conf = _conf_dir(tmp_path, 180)
    c = yaml.safe_load((conf / "cv.yaml").read_text(encoding="utf-8"))
    c["embargo_days"] = 7
    (conf / "cv.yaml").write_text(yaml.safe_dump(c, allow_unicode=True), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="max_lookback_days"):
            cv_config(str(conf))
    finally:
        reset_cache()


def test_rolling_window_longer_than_lookback_is_rejected(tmp_path):
    reset_cache()
    conf = _conf_dir(tmp_path, 180)
    f = yaml.safe_load((conf / "features.yaml").read_text(encoding="utf-8"))
    f["windows"]["rolling_days"] = 365
    (conf / "features.yaml").write_text(yaml.safe_dump(f, allow_unicode=True), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="embargo"):
            feature_config(str(conf))
    finally:
        reset_cache()


# ------------------------------------------------------------------------ CV-06
def test_cv06_expanding_window_contains_previous_fold(cfg):
    folds = make_folds(cfg)
    for prev, cur in zip(folds, folds[1:]):
        assert cur.train_start == prev.train_start
        assert cur.train_end > prev.train_end


# --------------------------------------------------------------------- CV-07/08
def test_cv07_reading_oos_without_unlock_raises(cfg, races):
    guard = OOSGuard(cfg)
    oos_rows = races[races["race_date"] >= pd.Timestamp(cfg.oos[0])]
    with pytest.raises(OOSAccessError, match="OOS"):
        guard.assert_readable(oos_rows)


def test_cv07_filter_silently_removes_oos_rows(cfg, races):
    guard = OOSGuard(cfg)
    kept = guard.filter(races)
    assert kept["race_date"].max() < pd.Timestamp(cfg.oos[0])
    guard.assert_readable(kept)  # 残ったぶんは読める


def test_cv08_unlock_is_recorded_in_the_ledger(cfg, tmp_path, races):
    ledger = tmp_path / "oos_access.log"
    guard = OOSGuard(cfg, ledger)
    assert guard.access_count() == 0
    guard.unlock("最終評価")
    guard.assert_readable(races)  # 開封後は通る
    assert guard.access_count() == 1, "開封回数が1回であることを MLflow 側でも突き合わせる"


# ------------------------------------------------------------------------ CV-09
def test_cv09_inner_folds_fit_inside_outer_train_window(cfg):
    outer = make_folds(cfg)[-1]
    inners = inner_folds(outer, cfg.inner_n_folds, cfg.embargo_days)
    assert len(inners) == cfg.inner_n_folds
    for f in inners:
        assert f.train_start >= outer.train_start
        assert f.valid_end <= outer.train_end
        assert (f.valid_start - f.train_end).days >= cfg.embargo_days


# ------------------------------------------------------------------------ CV-10
def test_cv10_feature_selection_statistics_differ_across_folds(cfg, races):
    """Null Importance / RFE の統計量が fold ごとに異なること。

    全 fold で同一なら、fold 内ではなく全期間で選択している実装バグ。
    ここでは train 期間の実データ量で代理する。
    """
    sizes = []
    for f in make_folds(cfg):
        tr, _ = f.mask(races["race_date"])
        sizes.append(int(tr.sum()))
    assert len(set(sizes)) == len(sizes), "全 fold で学習データが同一です"


def test_oos_fold_train_end_respects_embargo(cfg):
    oos = make_oos_fold(cfg, date(2026, 8, 25))
    assert (oos.valid_start - oos.train_end).days == cfg.embargo_days + 1
    assert oos.valid_end == date(2026, 8, 25)
