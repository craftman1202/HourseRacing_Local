"""人気との乖離（回帰）モデルの運用側（2026-10-05）。

確率変換の単一実装は学習側 nar.eval.deviation（learning/tests/test_deviation.py）。
ここでは読み込み・人気の作り方・推論経路を固定する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from narops.clock import FixedClock, jst_datetime
from narops.errors import ArtifactIntegrityError
from narops.inference import run_inference
from narops.model.manifest import Manifest
from narops.runtime import DeviationModel, load_deviation_model, needs_popularity
from narops.shared import popularity_from_odds

pytestmark = pytest.mark.unit
START = jst_datetime(2026, 10, 6, 15, 30)
TABLE = {"1_8": 0.40, "2_8": 0.20, "3_8": 0.12, "4_8": 0.08, "5_8": 0.06, "6_8": 0.05,
         "7_8": 0.05, "8_8": 0.04}


def _model(coef=(0.0, 0.0)):
    return DeviationModel(
        features=["f1", "f2"], booster=None, ridge_coef=np.array(coef, dtype=float),
        ridge_intercept=0.0,
        meta={"pop_table": TABLE, "reg_weights": {"ridge": 1.0},
              "reg_calibration": {"ridge": {"gamma": 0.5, "temperature": 1.0}}})


def _features():
    return pd.DataFrame({"horse_no": np.arange(1, 9), "f1": np.linspace(-1, 1, 8),
                         "f2": np.zeros(8)})


def test_needs_popularity_only_for_releases_with_market_pop_columns():
    assert needs_popularity(["h_winrate_prior", "market_pop_rank"])
    assert not needs_popularity(["h_winrate_prior", "d_all_starts"])


def test_popularity_is_ranked_from_pre_close_odds():
    assert popularity_from_odds(pd.Series([3.2, 1.5, 9.9])).tolist() == [2.0, 1.0, 3.0]


def test_with_zero_deviation_the_probability_is_the_popularity_prior():
    p = _model().predict(_features(), np.arange(1, 9, dtype=float), np.array(["R"] * 8))
    assert p.sum() == pytest.approx(1.0)
    expect = np.array([TABLE[f"{i}_8"] for i in range(1, 9)])
    assert np.allclose(p, expect / expect.sum())


def test_predicted_outperformance_raises_the_probability():
    # f1 が大きいほど「着順 − 人気」が小さい（人気より走る）と予測するモデル
    p = _model(coef=(-2.0, 0.0)).predict(_features(), np.full(8, 4.0), np.array(["R"] * 8))
    assert np.all(np.diff(p) > 0), "人気が同じなら、乖離予測が良い馬ほど確率が高いはず"


def test_missing_popularity_is_treated_as_last():
    p = _model().predict(_features(), np.full(8, np.nan), np.array(["R"] * 8))
    assert np.allclose(p, 1 / 8)


def test_run_inference_uses_the_deviation_model(release_dir, cfg):
    m = Manifest.read(Path(release_dir) / "manifest.json")
    odds = pd.Series([2.0, 3.5, 5.0, 8.0, 12.0, 20.0, 40.0, 80.0])
    res = run_inference(
        race_id="202026100601", features=_features(), models={}, manifests=[m],
        weights={}, temperature=1.0, clock=FixedClock(START - timedelta(minutes=10)),
        cfg=cfg, odds=odds, deviation=_model(),
        popularity=popularity_from_odds(odds).to_numpy())
    assert list(res.per_model) == ["deviation"]
    assert res.frame["p_win"].sum() == pytest.approx(1.0)
    assert res.frame["p_win"].iloc[0] == pytest.approx(0.40 / sum(TABLE.values()))


def _write_release(tmp_path, weights, with_lgbm=False):
    d = tmp_path / "rel"
    d.mkdir()
    (d / "feature_spec.json").write_text(json.dumps({"names": ["f1", "f2"]}))
    (d / "dev_reg_ridge.json").write_text(json.dumps({"coef": {"f1": 0.1, "f2": 0.2},
                                                      "intercept": 0.3}))
    (d / "dev_meta.json").write_text(json.dumps({
        "features": ["f1", "f2"], "pop_table": TABLE, "reg_weights": weights,
        "reg_calibration": {"ridge": {"gamma": 0.1, "temperature": 1.0}}}))

    @dataclass
    class Rel:
        path: Path
        release_id: str = "test"
    return Rel(d)


def test_loader_reads_ridge_and_meta(tmp_path):
    m = load_deviation_model(_write_release(tmp_path, {"ridge": 1.0}))
    assert m is not None and m.booster is None
    assert np.allclose(m.ridge_coef, [0.1, 0.2]) and m.ridge_intercept == 0.3


def test_loader_rejects_a_weighted_model_without_its_file(tmp_path):
    with pytest.raises(ArtifactIntegrityError):
        load_deviation_model(_write_release(tmp_path, {"ridge": 0.5, "lgbm": 0.5}))


def test_releases_without_dev_meta_have_no_deviation_model(release_dir):
    @dataclass
    class Rel:
        path: Path
        release_id: str = "x"
    assert load_deviation_model(Rel(Path(release_dir))) is None
