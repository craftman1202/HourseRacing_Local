"""fold チェックポイントと TabM の小さい探索空間（2026-10-03、HPO の OOM 対策）。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import optuna
import pandas as pd

from nar.eval.splits import Fold
from nar.train import hpo, pipeline as pl


def _fold():
    return Fold(index=3, train_start=date(1998, 1, 1), train_end=date(2020, 8, 4),
                valid_start=date(2021, 2, 1), valid_end=date(2021, 12, 31), embargo_days=180)


def _feat():
    return pd.DataFrame({"race_date": pd.to_datetime(["2020-01-01", "2021-03-01"]), "x": [1, 2]})


def _out():
    return pl.FoldOutput(fold=3, n_train=1, n_valid=1, selected_features=["x"],
                         predictions=pd.DataFrame(), metrics=pd.DataFrame(),
                         hpo={"lgbm": {"best_params": {"num_leaves": 31}}}, timings={},
                         temperatures={})


def test_checkpoint_roundtrip_and_invalidation(tmp_path):
    cfg = pl.RunConfig(hpo_trials={"lgbm": 30}, checkpoint_dir=str(tmp_path))
    f, feat = _fold(), _feat()
    assert pl._load_checkpoint(cfg, f, ["x"], feat) is None
    pl._save_checkpoint(cfg, f, ["x"], feat, _out())
    got = pl._load_checkpoint(cfg, f, ["x"], feat)
    assert got is not None and got.hpo["lgbm"]["best_params"]["num_leaves"] == 31
    # 設定・列・データのどれかが違えば使わない
    assert pl._load_checkpoint(replace(cfg, hpo_trials={"lgbm": 31}), f, ["x"], feat) is None
    assert pl._load_checkpoint(cfg, f, ["x", "y"], feat) is None
    assert pl._load_checkpoint(cfg, f, ["x"], feat.iloc[:1]) is None


def test_no_checkpoint_dir_means_no_files(tmp_path):
    cfg = pl.RunConfig()
    pl._save_checkpoint(cfg, _fold(), ["x"], _feat(), _out())
    assert pl._load_checkpoint(cfg, _fold(), ["x"], _feat()) is None
    assert not list(tmp_path.iterdir())


def test_small_tabm_space_fixes_k_at_four():
    study = optuna.create_study()
    for _ in range(10):
        t = study.ask()
        p = hpo.suggest_tabm_small(t)
        assert p["k"] == 4 and p["hidden"] in (128, 256)
        study.tell(t, 0.0)
    assert "tabm_small" in hpo.SUGGESTERS and hpo.SUGGESTERS["tabm"] is hpo.suggest_tabm
