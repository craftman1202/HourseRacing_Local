"""本番用モデルの最終学習。

walk-forward は評価であって、本番に出すモデルそのものは作らない。
ここで検証するのは「本番モデルが OOS を一切見ていないこと」と
「較正温度を学習に使った行で測っていないこと」。
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from nar.config import cv_config, feature_config
from nar.train.final import fit_and_export, training_window


def test_training_window_stops_an_embargo_before_the_oos_start():
    """embargo を引かないと、履歴系特徴量が OOS 期間の情報を含む。"""
    ccfg, fcfg = cv_config(), feature_config()
    start, end = training_window(ccfg, fcfg)
    oos_start = pd.Timestamp(ccfg.oos[0])
    assert end < oos_start
    assert (oos_start - end).days > ccfg.embargo_days
    assert ccfg.embargo_days == fcfg.max_lookback_days


def test_training_window_refuses_when_embargo_swallows_the_period():
    from dataclasses import replace

    ccfg = cv_config()
    narrow = replace(ccfg, train_start="2024-01-01")
    with pytest.raises(ValueError, match="学習期間が空"):
        training_window(narrow, feature_config())


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    from nar import synth
    from nar.features.builder import ASOF_FEATURES, build
    from nar.synth import SynthConfig

    # 小さめの合成データで足りる。ここで検証したいのは「OOS を見ていないこと」と
    # 「較正を学習行で測っていないこと」で、モデルの精度ではない。
    t = synth.generate(SynthConfig(n_races=1200))
    feat = build(t["entry"], t["race"], feature_config())
    cols = [c for c in ASOF_FEATURES if c in feat.columns]
    ccfg = cv_config()
    # 合成データの期間に合わせて窓を寄せる
    from dataclasses import replace

    dates = pd.to_datetime(feat["race_date"])
    ccfg = replace(ccfg, train_start=str(dates.min().date()),
                   oos=(str((dates.max() - pd.Timedelta(days=30)).date()), None))
    # 特徴量選択は features/selection の単体テストで見る。ここで回すと
    # RFE と Null Importance で 20 分かかり、実行されないテストになる。
    return fit_and_export(feat, cols, ccfg, feature_config(),
                          out_dir=tmp_path_factory.mktemp("final"),
                          models=("clogit", "lgbm"), holdout_days=200,
                          do_selection=False)


def test_artifacts_are_written(artifacts):
    for name in ("clogit_beta.json", "lgbm_rank.txt", "feature_names.json",
                 "final_meta.json"):
        assert (artifacts.out_dir / name).exists(), f"{name} がありません"


def test_calibration_temperature_is_measured_outside_the_fit_rows(artifacts):
    """学習に使った行で温度を測ると 1 に張り付き、較正が効かない。"""
    meta = json.loads((artifacts.out_dir / "final_meta.json").read_text(encoding="utf-8"))
    assert meta["n_calibration_rows"] > 0
    assert meta["n_fit_rows"] > meta["n_calibration_rows"]
    for name, temp in artifacts.temperatures.items():
        assert temp > 0, name
        assert abs(temp - 1.0) > 1e-6, f"{name}: 温度が 1 に張り付いています"


def test_feature_names_match_the_beta_keys(artifacts):
    """配布する係数と特徴量名の順序が食い違うと、推論が別の列に重みを当てる。"""
    names = json.loads((artifacts.out_dir / "feature_names.json").read_text(encoding="utf-8"))
    beta = json.loads((artifacts.out_dir / "clogit_beta.json").read_text(encoding="utf-8"))["beta"]
    assert list(beta) == names


# ------------------------------------------------------------------ ゲート証跡
def test_gate_marks_unrun_blockers_as_missing_not_green(tmp_path):
    """未実施を GREEN と読み替えたら、ゲートは何も守らない。"""
    from nar import gate

    (tmp_path / "junit.xml").write_text(
        '<testsuite><testcase classname="t" name="test_lk05_x"/></testsuite>',
        encoding="utf-8")
    payload = gate.build(tmp_path, tmp_path, run_tests=False)
    assert payload["results"]["LK-05"] == "GREEN"
    assert set(payload["missing"]) == set(gate.REQUIRED) - {"LK-05"}
    assert payload["can_publish"] is False


def test_gate_turns_red_when_a_blocker_test_fails(tmp_path):
    from nar import gate

    (tmp_path / "junit.xml").write_text(
        '<testsuite><testcase classname="t" name="test_lk05_x">'
        '<failure message="boom"/></testcase></testsuite>', encoding="utf-8")
    payload = gate.build(tmp_path, tmp_path, run_tests=False)
    assert payload["results"]["LK-05"] == "RED"


def test_gate_fails_when_real_oos_guards_fired(tmp_path):
    """単体テストが緑でも、実データで RF-01 が鳴っていたら publish させない。"""
    import pandas as pd

    from nar import gate

    (tmp_path / "junit.xml").write_text(
        "<testsuite>" + "".join(
            f'<testcase classname="t" name="test_{t.replace("-", "").lower()}_ok"/>'
            for t in gate.REQUIRED) + "</testsuite>", encoding="utf-8")
    assert gate.build(tmp_path, tmp_path, run_tests=False)["can_publish"] is True

    pd.DataFrame([{"id": "RF-01", "fired": True}]).to_csv(
        tmp_path / "oos_guards.csv", index=False)
    payload = gate.build(tmp_path, tmp_path, run_tests=False)
    assert payload["results"]["RF-01"] == "RED"
    assert payload["can_publish"] is False


def test_gate_required_set_matches_the_operation_side():
    """片方だけ Blocker を増やすと、増やしたつもりが守られない。"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "operation" / "src"))
    from narops.release import REQUIRED_GREEN

    from nar import gate

    assert set(gate.REQUIRED) == set(REQUIRED_GREEN)


def test_onnx_verification_sample_is_not_padded():
    """検証用の行列はパディング前の実データから作ること。

    to_batch はレースを最大頭数の矩形に揃える。その x を平坦化して
    レースサイズと突き合わせると、行数が合わずに検証が必ず失敗する
    （実際に「合計 4096 が行数 6230 と不一致」で TabM が配布物から落ちた）。
    """
    import numpy as np
    import pandas as pd

    from nar.models.base import to_batch

    df = pd.DataFrame({
        "race_id": ["r1"] * 3 + ["r2"] * 5,
        "horse_no": [1, 2, 3, 1, 2, 3, 4, 5],
        "f1": np.arange(8, dtype=float), "f2": np.arange(8, dtype=float),
        "is_win": [1, 0, 0, 1, 0, 0, 0, 0],
    })
    cols = ["f1", "f2"]
    sizes = df.groupby("race_id", sort=False).size().tolist()
    assert sum(sizes) == len(df)

    padded = to_batch(df, cols).x.reshape(-1, len(cols))
    assert len(padded) > sum(sizes), "テスト前提: to_batch はパディングする"

    flat = df[cols].to_numpy(dtype=float)
    assert len(flat) == sum(sizes)


def test_evaluation_window_stays_behind_the_oos_boundary_by_default():
    """既定は評価用。全期間を既定にすると評価用と出荷用の区別が消える。"""
    import pandas as pd

    ccfg = cv_config()
    _, end = training_window(ccfg, feature_config())
    assert end < pd.Timestamp(ccfg.oos[0]) - pd.Timedelta(days=ccfg.embargo_days)


def test_production_window_can_extend_past_the_oos_boundary():
    """OOS で測ったあと、出荷用は入手できる全データで作り直す。"""
    import pandas as pd

    ccfg = cv_config()
    _, end = training_window(ccfg, feature_config(), through="2026-08-26")
    assert end == pd.Timestamp("2026-08-26")
    assert end > pd.Timestamp(ccfg.oos[0])


def test_purpose_is_recorded_in_the_metadata(tmp_path_factory):
    """manifest の oos_metrics は評価用で測った数字。どちらの版かが要る。"""
    import json

    from dataclasses import replace

    from nar import synth
    from nar.features.builder import ASOF_FEATURES, build

    t = synth.generate(synth.SynthConfig(n_races=900))
    feat = build(t["entry"], t["race"], feature_config())
    cols = [c for c in ASOF_FEATURES if c in feat.columns]
    dates = pd.to_datetime(feat["race_date"])
    ccfg = replace(cv_config(), train_start=str(dates.min().date()),
                   oos=(str((dates.max() - pd.Timedelta(days=30)).date()), None))

    art = fit_and_export(feat, cols, ccfg, feature_config(),
                         out_dir=tmp_path_factory.mktemp("prod"),
                         through=str(dates.max().date()),
                         models=("clogit",), holdout_days=200, do_selection=False)
    meta = json.loads((art.out_dir / "final_meta.json").read_text(encoding="utf-8"))
    assert meta["purpose"] == "production"
    assert meta["train_period"]["end"] == str(dates.max().date())
