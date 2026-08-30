"""ばんえい variant の特徴量。

ばんえいは 200m 直線・そりの重量で決まる別競技で、平地の 20 特徴量のうち
距離・回り・自己ベストタイム由来の列は情報を持たない（features/banei.py の
docstring に実データの根拠）。ここで固定するのは3点:

  1. variant を切り替えても平地側の列集合が1文字も変わらないこと
  2. ばんえい側で使い物にならない列が物理的に落ちていること
  3. 重量まわりのパースが実データの表記（☆620 / +12 / ±0）を取り違えないこと
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nar.config import CONF_DIR, feature_config
from nar.features import banei
from nar.features.banei import BANEI_FEATURES, MOISTURE_MAX
from nar.features.builder import ASOF_FEATURES, BANEI_DROPPED, asof_features

BANEI_CONF = str(CONF_DIR.parent / "conf_banei")


@pytest.fixture(scope="module")
def bcfg():
    return feature_config(BANEI_CONF)


# ------------------------------------------------------------------ 設定
def test_banei_conf_selects_all_four_banei_tracks(bcfg):
    """ばんえいは帯広だけではない。1998-2006 に 北見ば/岩見ば/旭川ば が実在する。"""
    assert bcfg.variant == "banei"
    assert sorted(bcfg.include_baba_codes) == [1, 2, 3, 4]
    assert bcfg.exclude_baba_codes == []


def test_flat_conf_is_untouched():
    """平地の設定は 1 行も変わっていない（ばんえいを除外したまま）。"""
    f = feature_config()
    assert f.variant == "flat"
    assert sorted(f.exclude_baba_codes) == [1, 2, 3, 4]
    assert f.include_baba_codes == []


def test_include_and_exclude_may_not_overlap(tmp_path):
    """どちらが効くか読めない設定は受け付けない。"""
    import shutil

    import yaml

    from nar.config import reset_cache

    shutil.copytree(CONF_DIR, tmp_path / "conf")
    raw = yaml.safe_load((tmp_path / "conf" / "features.yaml").read_text(encoding="utf-8"))
    raw["include"] = {"baba_codes": [3]}
    raw["exclude"] = {"baba_codes": [3]}
    (tmp_path / "conf" / "features.yaml").write_text(
        yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    reset_cache()
    with pytest.raises(ValueError, match="include と exclude"):
        feature_config(str(tmp_path / "conf"))
    reset_cache()


# ------------------------------------------------------------------ 列集合
def test_flat_feature_list_is_bit_identical_to_the_module_constant():
    """variant を足したことで平地の列や順序が動いていないこと。

    順序は feature_spec のハッシュに乗る。ここがずれると配布済みモデルが
    「特徴量仕様の不一致」で止まる。
    """
    assert asof_features(None) == ASOF_FEATURES
    assert asof_features(feature_config()) == ASOF_FEATURES


def test_banei_feature_list_drops_the_degenerate_columns(bcfg):
    cols = asof_features(bcfg)
    for c in BANEI_DROPPED:
        assert c not in cols, f"{c} はばんえいでは情報を持たない（落とすべき）"
    assert "distance" not in cols, "全レース 200m 固定なので分散ゼロ"
    assert set(BANEI_FEATURES) <= set(cols)
    # 平地由来で残るべきものは残っている
    for c in ("h_winrate_prior", "j_winrate_shrunk", "class_level", "d_all_starts"):
        assert c in cols


def test_banei_feature_names_are_unique_and_ordered(bcfg):
    cols = asof_features(bcfg)
    assert len(cols) == len(set(cols))
    assert cols[-len(BANEI_FEATURES):] == BANEI_FEATURES


# ------------------------------------------------------------------ パース
def _frame(**over) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(over.get("weight_carried", ["600"]))
    base = {
        "race_id": ["R1"] * n, "horse_no": list(range(1, n + 1)),
        "weight_carried": ["600"] * n, "weight_kg": [1000.0] * n,
        "weight_diff": ["+2"] * n, "sex": ["牡"] * n, "age": [5] * n,
    }
    base.update(over)
    race = pd.DataFrame([{"race_id": "R1", "baba_condition": over.get("_cond", "1.8")}])
    return pd.DataFrame(base), race


def test_apprentice_mark_is_stripped_from_the_weight_and_kept_as_a_flag():
    """実データの負担重量は `☆620` のように減量印が前置される（5万行以上）。"""
    e, r = _frame(weight_carried=["620", "☆620"])
    out = banei.build(e, r)
    assert out["b_weight_carried"].tolist() == [620.0, 620.0]
    assert out["b_is_apprentice"].tolist() == [0.0, 1.0]


def test_weight_rel_is_the_within_race_handicap_difference():
    """ばんえいの重量はハンデそのもの。効くのは絶対値よりレース内の差。"""
    e, r = _frame(weight_carried=["600", "620", "580", "600"])
    out = banei.build(e, r)
    assert out["b_weight_rel"].tolist() == [0.0, 20.0, -20.0, 0.0]
    assert out["b_weight_rel"].sum() == pytest.approx(0.0)


def test_body_weight_diff_keeps_its_sign():
    e, r = _frame(weight_carried=["600"] * 3, weight_diff=["+12", "-14", "±0"])
    out = banei.build(e, r)
    assert out["b_body_weight_diff"].tolist() == [12.0, -14.0, 0.0]


def test_load_ratio_is_the_sled_weight_over_the_body_weight():
    e, r = _frame(weight_carried=["600"], weight_kg=[1000.0])
    assert banei.build(e, r)["b_load_ratio"].iloc[0] == pytest.approx(0.6)


def test_missing_body_weight_stays_missing_rather_than_zero():
    """0 で埋めると load_ratio が無限大になる。欠測は欠測のまま通す。"""
    e, r = _frame(weight_carried=["600"], weight_kg=[np.nan])
    out = banei.build(e, r)
    assert np.isnan(out["b_body_weight"].iloc[0])
    assert np.isnan(out["b_load_ratio"].iloc[0])


def test_moisture_reads_the_numeric_track_condition():
    e, r = _frame(weight_carried=["600"], _cond="1.8")
    assert banei.build(e, r)["b_moisture"].iloc[0] == pytest.approx(1.8)


def test_implausible_moisture_is_dropped_not_zero_filled():
    """実データの 2004 年だけ 60-69 が 766 行ある。砂は飽和しても 20% 程度。

    0 で埋めると「乾いた馬場」に化けて意味が反転するので、欠測にする。
    """
    e, r = _frame(weight_carried=["600"], _cond=str(MOISTURE_MAX + 1))
    assert np.isnan(banei.build(e, r)["b_moisture"].iloc[0])


def test_categorical_track_condition_is_not_forced_into_a_number():
    """平地の 良/稍重 がばんえい特徴量に紛れ込んでも数値化しない。"""
    e, r = _frame(weight_carried=["600"], _cond="良")
    assert np.isnan(banei.build(e, r)["b_moisture"].iloc[0])


def test_gelding_is_recognised_in_both_notations():
    """実データには「セン」と「去」の両方が存在する。"""
    e, r = _frame(weight_carried=["600"] * 4, sex=["牡", "牝", "セン", "去"])
    out = banei.build(e, r)
    assert out["b_is_female"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert out["b_is_gelding"].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_all_banei_features_are_float64_even_when_empty():
    """列の dtype がデータ依存だと、学習時と推論時で feature_spec がずれる。"""
    e, r = _frame(weight_carried=["600"])
    out = banei.build(e.drop(columns=["weight_kg", "sex", "age"]), r)
    for c in BANEI_FEATURES:
        assert str(out[c].dtype) == "float64", c


# ------------------------------------------------------------------ LK-05
def _banei_like(n_races: int = 400, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ばんえいに似せた合成データ。

    `nar.synth` は平地向けなので（馬体重 460kg・回りあり・馬場は 良/稍重）、
    重量とばんえいの馬場表記を持つデータをここで作る。目的は分布の再現ではなく、
    ばんえい variant の全経路に as-of 違反が無いことの確認。
    """
    rng = np.random.default_rng(seed)
    n_horses, n_jockeys = 220, 25
    horse_ability = rng.normal(0, 1, n_horses)
    days = pd.date_range("2020-01-01", periods=n_races // 4 + 1, freq="7D")

    races, entries = [], []
    for i in range(n_races):
        day = days[i % len(days)]
        race_no = i // len(days) + 1
        race_id = f"03{day.strftime('%Y%m%d')}{race_no:02d}"
        start_ts = day + pd.Timedelta(minutes=600 + race_no * 30)
        n = int(rng.integers(8, 11))
        h = rng.choice(n_horses, size=n, replace=False)
        j = rng.choice(n_jockeys, size=n, replace=False)
        carried = 600 + rng.integers(-4, 5, n) * 10
        util = horse_ability[h] - 0.02 * (carried - carried.mean()) + rng.gumbel(0, 1, n)
        finish = np.empty(n, dtype=int)
        finish[np.argsort(-util)] = np.arange(1, n + 1)
        races.append({
            "race_id": race_id, "race_date": day.date(), "start_ts": start_ts,
            "baba_code": 3, "race_no": race_no, "distance": 200, "surface": "ば",
            "turn": "直", "class_level": int(rng.choice([2, 4, 5])),
            "baba_condition": f"{rng.uniform(0.5, 6.0):.1f}",
            "prize_yen": int(rng.lognormal(13.0, 0.3)), "n_runners": n,
        })
        for k in range(n):
            mark = "☆" if rng.random() < 0.06 else ""
            entries.append({
                "race_id": race_id, "race_date": day.date(), "start_ts": start_ts,
                "baba_code": 3, "distance": 200, "horse_no": k + 1, "waku": k // 2 + 1,
                "horse_sk": f"H{h[k]:04d}", "jockey_sk": f"J{j[k]:03d}",
                "trainer_sk": f"T{h[k] % 30:03d}", "sire_sk": f"S{h[k] % 15:03d}",
                "finish_pos": int(finish[k]), "is_win": int(finish[k] == 1),
                "time_sec": float(100 + finish[k] * 2 + rng.normal(0, 4)),
                "weight_carried": f"{mark}{carried[k]}",
                "weight_kg": float(rng.normal(1000, 60)),
                "weight_diff": f"{'+' if rng.random() < 0.5 else '-'}{rng.integers(0, 20)}",
                "sex": str(rng.choice(["牡", "牝", "セン"], p=[0.68, 0.29, 0.03])),
                "age": int(rng.integers(2, 12)),
            })
    return pd.DataFrame(entries), pd.DataFrame(races)


def test_lk05_banei_variant_leaves_past_features_bit_identical(bcfg):
    """T 以降を乱数で破壊しても、T 以前のばんえい特徴量が1ビットも変わらないこと。

    平地と同じ最重要不変条件（LK-05）。重量・含水率は行ごとに閉じた値なので
    原理的に未来を見ないが、経路全体を通して確認する。
    """
    from nar.eda import leakage
    from nar.features.builder import build, content_hash

    entry, race = _banei_like()
    cut = pd.Timestamp("2021-06-01")

    rng = np.random.default_rng(11)
    poisoned = entry.copy()
    future = pd.to_datetime(poisoned["start_ts"]) >= cut
    poisoned.loc[future, "finish_pos"] = rng.integers(1, 11, int(future.sum()))
    poisoned.loc[future, "is_win"] = rng.integers(0, 2, int(future.sum()))
    poisoned.loc[future, "time_sec"] = rng.normal(120, 20, int(future.sum()))
    poisoned.loc[future, "weight_carried"] = "999"
    poisoned.loc[future, "weight_kg"] = 1200.0

    past = lambda f: f[f["start_ts"] < cut].sort_values(
        ["race_id", "horse_no"]).reset_index(drop=True)
    p1 = past(build(entry, race, bcfg))
    p2 = past(build(poisoned, race, bcfg))

    assert len(p1) > 0
    assert len(leakage.future_poisoning_diff(p1, p2)) == 0
    assert content_hash(p1) == content_hash(p2)


def test_banei_build_excludes_flat_tracks(bcfg):
    """include=[1,2,3,4] は「それ以外を入れない」の意味。

    平地が混ざると騎手・調教師の勝率が別競技の成績で薄まる。
    """
    from nar.features.builder import build

    entry, race = _banei_like(n_races=120, seed=3)
    flat_e, flat_r = entry.copy(), race.copy()
    flat_e["baba_code"] = 20
    flat_r["baba_code"] = 20
    flat_e["race_id"] = "20" + flat_e["race_id"].str[2:]
    flat_r["race_id"] = "20" + flat_r["race_id"].str[2:]

    out = build(pd.concat([entry, flat_e], ignore_index=True),
                pd.concat([race, flat_r], ignore_index=True), bcfg)
    assert (out["baba_code"] == 3).all(), "平地の行がばんえいの学習集合に残っています"


# ------------------------------------------------------------------ gold の分離
def test_gold_subdir_separates_the_two_variants(bcfg):
    """平地とばんえいが同じ features.parquet を奪い合わないこと。

    ここが共通だと、後から走らせた方が相手の gold を黙って上書きし、
    fit-final が「別競技の特徴量で学習したモデル」を出す。
    """
    from nar.cli import _gold_subdir

    flat = feature_config()
    assert _gold_subdir(flat) == "features_noodds"
    assert _gold_subdir(bcfg) == "features_noodds_banei"
    assert _gold_subdir(flat) != _gold_subdir(bcfg)


def test_no_flat_feature_starts_with_the_banei_prefix():
    """系統の判定は特徴量名の `b_` 接頭辞で行う（narops.model.registry.family_of）。

    平地の列がひとつでも `b_` で始まると、平地のリリースがばんえいと誤判定され、
    current の取り違え検査が逆に働く。
    """
    offenders = [c for c in ASOF_FEATURES if c.startswith("b_")]
    assert offenders == [], (
        f"平地の特徴量に `b_` 接頭辞があります: {offenders}。"
        "narops.model.registry._BANEI_PREFIX と衝突します。")


def test_every_banei_feature_uses_the_prefix():
    assert all(c.startswith("b_") for c in BANEI_FEATURES)


# ------------------------------------------------------- ゲートの実行環境
def test_gate_pytest_does_not_inherit_the_variant_conf(monkeypatch, tmp_path):
    """`NAR_CONF_DIR` を子プロセスに渡さないこと。

    ばんえいの学習から `NAR_CONF_DIR=conf_banei nar gate-report` と呼ぶと、
    pytest がそれを受け継いで全テストがばんえいの設定で走る。実際に 16 件が
    落ちて LK-05/LK-06 が RED になり、publish が止まった（2026-08-30）。
    ゲートが問うのは「このコードで Blocker が通るか」であって、
    どの variant を出荷しようとしているかではない。
    """
    import subprocess

    from nar import gate

    seen = {}

    def fake_run(cmd, check=False, env=None):
        seen["env"] = env
        Path(cmd[-1].split("=", 1)[1]).write_text(
            "<testsuites><testsuite></testsuite></testsuites>", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setenv("NAR_CONF_DIR", "conf_banei")
    monkeypatch.setenv("NAR_SOMETHING_ELSE", "keep-me")
    monkeypatch.setattr(gate.subprocess, "run", fake_run)

    gate.run_pytest(tmp_path / "tests", tmp_path / "junit.xml")

    assert "NAR_CONF_DIR" not in seen["env"], (
        "variant の設定が pytest に漏れています")
    assert seen["env"]["NAR_SOMETHING_ELSE"] == "keep-me", (
        "関係のない環境変数まで落としています")
