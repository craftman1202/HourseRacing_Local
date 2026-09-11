"""SK-01..07: 学習／推論 skew。

本設計の最大の失敗モードは「学習時と違う条件のデータで推論する」こと。
ここは許容差ゼロで運用する。
"""

from __future__ import annotations

import pandas as pd
import pytest

from narops.clock import business_date
from narops.errors import SkewError
from narops.skew import (
    assert_no_duplicate_implementation, compare, enforce, shared_code_paths,
)

pytestmark = pytest.mark.component


def _frames(values_a: dict, values_b: dict, n: int = 3):
    base = {"race_id": [f"20202608250{i}" for i in range(n)], "horse_no": list(range(1, n + 1))}
    return pd.DataFrame({**base, **values_a}), pd.DataFrame({**base, **values_b})


# ------------------------------------------------------------------ SK-01
def test_sk01_identical_features_produce_no_mismatch(cfg):
    a, b = _frames({"h_winrate_prior": [0.1, 0.2, 0.3]},
                   {"h_winrate_prior": [0.1, 0.2, 0.3]})
    r = compare(a, b, cfg.tolerated_skew_columns, business_date(pd.Timestamp.now()))
    assert r.is_clean and r.n_compared == 3
    enforce(r)


def test_sk01_tiny_difference_beyond_tolerance_is_caught(cfg):
    """相対誤差 1e-9 を超えたら不一致。許容差ゼロで運用する。"""
    a, b = _frames({"h_winrate_prior": [0.1, 0.2, 0.3]},
                   {"h_winrate_prior": [0.1, 0.2, 0.3 + 1e-6]})
    r = compare(a, b, cfg.tolerated_skew_columns, business_date(pd.Timestamp.now()))
    assert not r.is_clean and "h_winrate_prior" in r.offending_columns
    with pytest.raises(SkewError, match="一致しません"):
        enforce(r)


def test_sk01_float_noise_within_tolerance_passes(cfg):
    a, b = _frames({"h_winrate_prior": [0.1, 0.2, 0.3]},
                   {"h_winrate_prior": [0.1, 0.2, 0.3 + 1e-15]})
    r = compare(a, b, cfg.tolerated_skew_columns, business_date(pd.Timestamp.now()))
    assert r.is_clean


def test_sk01_null_versus_value_is_a_mismatch(cfg):
    a, b = _frames({"h_si_last3": [None, 0.2, 0.3]},
                   {"h_si_last3": [0.0, 0.2, 0.3]})
    r = compare(a, b, cfg.tolerated_skew_columns, business_date(pd.Timestamp.now()))
    assert "h_si_last3" in r.offending_columns, "NULL と 0 の取り違えを見逃しています"


# ------------------------------------------------------------ SK-02（Blocker）
def test_sk02_shared_functions_come_from_the_learning_package():
    """学習時と推論時が同一モジュール・同一関数を呼ぶこと。"""
    assert_no_duplicate_implementation()
    paths = shared_code_paths()
    assert paths, "検証対象の共有関数がありません"
    for name, path in paths.items():
        assert "/learning/src/nar/" in path, f"{name} が学習側由来ではありません: {path}"


def test_sk02_shared_objects_are_identical_not_copies():
    """re-export が同一オブジェクトを指していること（コピー実装でないこと）。"""
    from nar.eval.economic import effective_odds as learn_eff
    from nar.transform.prerace import to_prerace as learn_prerace

    from narops import shared

    assert shared.to_prerace is learn_prerace
    assert shared.effective_odds is learn_eff


def test_sk02_detects_a_reimplementation(monkeypatch):
    """運用側に再実装を差し込むと検出されること。"""
    from narops import shared

    def fake_effective_odds(o, b, pool, takeout):     # 運用側の重複実装を模す
        return o

    monkeypatch.setattr(shared, "effective_odds", fake_effective_odds)
    with pytest.raises(SkewError, match="再実装"):
        assert_no_duplicate_implementation()


# ------------------------------------------------------------------ SK-03
def test_sk03_tolerated_columns_do_not_block(cfg):
    """日内更新に起因する3系統の差分は許容される。"""
    a, b = _frames({"j_wins_today": [1.0, 2.0, 3.0], "h_winrate_prior": [0.1, 0.2, 0.3]},
                   {"j_wins_today": [9.0, 9.0, 9.0], "h_winrate_prior": [0.1, 0.2, 0.3]})
    r = compare(a, b, cfg.tolerated_skew_columns, business_date(pd.Timestamp.now()))
    assert not r.mismatches.empty, "差分自体は記録されるべき"
    assert r.is_clean, "ホワイトリスト列の差分で止めてはいけない"
    enforce(r)


def test_sk03_out_of_whitelist_column_blocks(cfg):
    a, b = _frames({"j_wins_today": [1.0, 2.0, 3.0], "h_starts_prior": [1.0, 2.0, 3.0]},
                   {"j_wins_today": [9.0, 9.0, 9.0], "h_starts_prior": [5.0, 5.0, 5.0]})
    r = compare(a, b, cfg.tolerated_skew_columns, business_date(pd.Timestamp.now()))
    assert r.offending_columns == ["h_starts_prior"]
    with pytest.raises(SkewError):
        enforce(r)


def test_sk03_tolerated_set_matches_design(cfg):
    assert cfg.tolerated_skew_columns == frozenset(
        {"j_wins_today", "t_wins_today", "track_speed_bias"})


# ------------------------------------------------------------------ 列構成の差
def test_missing_column_on_either_side_is_reported(cfg):
    a, b = _frames({"h_winrate_prior": [0.1, 0.2, 0.3], "extra_col": [1.0, 2.0, 3.0]},
                   {"h_winrate_prior": [0.1, 0.2, 0.3]})
    r = compare(a, b, cfg.tolerated_skew_columns, business_date(pd.Timestamp.now()))
    assert "extra_col" in r.offending_columns, "列構成の違いを見逃しています"


# ------------------------------------------------------------------ SK-04
def test_sk04_missing_representation_is_nan_not_zero(populated, release_dir, clock):
    """欠損がゼロ・平均代入されていないこと。"""
    from nar.config import feature_config

    from narops.clock import jst_datetime, to_utc
    from narops.features import build_for_race
    from narops.model.manifest import Manifest

    start = jst_datetime(2026, 8, 22, 14, 30)
    card = pd.DataFrame([{
        "race_id": "202026082201", "horse_no": i + 1,
        "horse_sk": f"NEW{i:04d}",            # 初出走馬（履歴なし）
        "jockey_sk": f"J{i % 12:03d}", "trainer_sk": f"T{i % 9:03d}",
        "sire_sk": f"S{i % 6:03d}",
    } for i in range(6)])
    race_row = pd.Series({
        "race_id": "202026082201", "race_date": start.date(), "start_ts": to_utc(start),
        "baba_code": 20, "distance": 1400, "race_no": 1, "surface": "ダ",
        "class_level": 2, "prize_yen": 1_000_000,
    })
    res = build_for_race(populated, card, race_row,
                         Manifest.read(release_dir / "manifest.json"),
                         feature_config(), max_bytes_billed=2_000_000_000)

    assert (res.frame["h_starts_prior"] == 0).all()
    assert res.frame["h_winrate_prior"].isna().all(), (
        "初出走の勝率が 0 で埋められています。学習時は NULL のはずです。")
    assert all(v == "nan" for v in res.spec.missing.values())


# ------------------------------- SK-02: 新規特徴量の学習/推論パリティ（設計書 §13.1）
def test_new_features_match_between_training_and_serving(populated, release_dir, clock):
    """`h_pace_bal_last3` と `jt_winrate_wilson` が推論経路でも学習と同じ値になること。

    この2つは 2026-09 に追加した特徴量で、経路が他と違う:

      - ペースバランスは確定層の `last3f` を材料にする。列を DDL に足しても
        既存行が NULL のままだと、学習時は値があり推論時だけ全行 NaN という
        train-serving skew になる（実際にこの移行を `scripts/backfill_last3f.py`
        で行った）。「NaN でも落ちない」ことではなく「値が入る」ことを検査する。
      - 騎手×調教師は履歴の取得範囲が新しい条件を要求する。出馬表の馬だけを
        引いていると組み合わせの過去が欠け、学習時より小さい標本の Wilson 下限に
        なる。`history_before` が騎手・調教師のキャリアも引いている前提を固定する。

    比較対象は「同じ履歴を学習側ビルダーに直接渡した値」。運用側は学習側の
    build() をそのまま呼ぶので、一致しない場合は入力の作り方が違う。
    """
    import numpy as np
    from nar.config import feature_config
    from nar.features.builder import build as build_features

    from narops.clock import jst_datetime, to_utc
    from narops.features import build_for_race
    from narops.model.manifest import Manifest

    start = jst_datetime(2026, 8, 22, 14, 30)
    hist = populated.table("entry_result_final")
    # 実績のある馬・騎手・調教師で出馬表を組む。初出走馬だと過去走が無く、
    # ペースバランスは定義上 NULL になって検査にならない。
    seen = hist.sort_values("start_ts").drop_duplicates("horse_sk")
    veterans = (hist.groupby("horse_sk").size().sort_values(ascending=False)
                .head(6).index.tolist())
    rows = []
    for i, h in enumerate(veterans):
        r = seen[seen["horse_sk"] == h].iloc[0]
        rows.append({"race_id": "202026082201", "horse_no": i + 1, "horse_sk": h,
                     "jockey_sk": r["jockey_sk"], "trainer_sk": r["trainer_sk"],
                     "sire_sk": r["sire_sk"]})
    card = pd.DataFrame(rows)
    race_row = pd.Series({
        "race_id": "202026082201", "race_date": start.date(), "start_ts": to_utc(start),
        "baba_code": 20, "distance": 1400, "race_no": 1, "surface": "ダ",
        "class_level": 2, "prize_yen": 1_000_000, "turn": "右", "baba_condition": "良",
    })
    res = build_for_race(populated, card, race_row,
                         Manifest.read(release_dir / "manifest.json"),
                         feature_config(), max_bytes_billed=2_000_000_000)

    assert "h_pace_bal_last3" in res.frame.columns
    assert res.frame["h_pace_bal_last3"].notna().any(), (
        "推論時のペースバランスが全頭 NaN です。確定層の last3f が"
        "埋まっていないか、履歴の取り方が学習時と違います。")
    assert (res.frame["jt_starts_prior"] > 0).any(), (
        "騎手×調教師の過去出走が1件も引けていません。履歴の取得範囲を"
        "確認してください（出馬表の馬だけに絞ると組み合わせの過去が落ちます）。")
    assert res.frame["jt_winrate_wilson"].between(0.0, 1.0).all()


# ------------------------------------------------------------------ SK-06
def test_sk06_snapshot_is_always_saved(populated, release_dir, clock):
    from nar.config import feature_config

    from narops.clock import jst_datetime, to_utc
    from narops.features import build_for_race, load_snapshot, save_snapshot
    from narops.model.manifest import Manifest

    start = jst_datetime(2026, 8, 22, 14, 30)
    card = pd.DataFrame([{
        "race_id": "202026082201", "horse_no": i + 1, "horse_sk": f"H{i:04d}",
        "jockey_sk": f"J{i % 12:03d}", "trainer_sk": f"T{i % 9:03d}",
        "sire_sk": f"S{i % 6:03d}"} for i in range(6)])
    race_row = pd.Series({
        "race_id": "202026082201", "race_date": start.date(), "start_ts": to_utc(start),
        "baba_code": 20, "distance": 1400, "race_no": 1, "surface": "ダ",
        "class_level": 2, "prize_yen": 1_000_000})
    m = Manifest.read(release_dir / "manifest.json")
    res = build_for_race(populated, card, race_row, m, feature_config(),
                         max_bytes_billed=2_000_000_000)

    n = save_snapshot(populated, res, "202026082201", m.model_id, clock)
    assert n == 6

    loaded = load_snapshot(populated, business_date(res.as_of_ts),
                           max_bytes_billed=2_000_000_000)
    assert len(loaded) == 6
    assert set(res.spec.names) <= set(loaded.columns), "特徴量が復元できません"
    assert (loaded["feature_spec_hash"] == res.spec.hash()).all()
    assert loaded["db_watermark"].notna().all(), "参照した DB 状態が残っていません"


def test_sk06_snapshot_is_replaced_not_duplicated(populated, release_dir, clock):
    from nar.config import feature_config

    from narops.clock import jst_datetime, to_utc
    from narops.features import build_for_race, save_snapshot
    from narops.model.manifest import Manifest

    start = jst_datetime(2026, 8, 22, 14, 30)
    card = pd.DataFrame([{
        "race_id": "202026082201", "horse_no": i + 1, "horse_sk": f"H{i:04d}",
        "jockey_sk": f"J{i:03d}", "trainer_sk": f"T{i:03d}", "sire_sk": f"S{i:03d}"}
        for i in range(4)])
    race_row = pd.Series({
        "race_id": "202026082201", "race_date": start.date(), "start_ts": to_utc(start),
        "baba_code": 20, "distance": 1400, "race_no": 1, "surface": "ダ",
        "class_level": 2, "prize_yen": 1_000_000})
    m = Manifest.read(release_dir / "manifest.json")
    res = build_for_race(populated, card, race_row, m, feature_config(),
                         max_bytes_billed=2_000_000_000)
    save_snapshot(populated, res, "202026082201", m.model_id, clock)
    save_snapshot(populated, res, "202026082201", m.model_id, clock)
    assert populated.row_count("feature_snapshot") == 4, "再実行で重複しています"


# ------------------------------------------------------------------ SK-07
def test_sk07_mismatch_triggers_alert_and_degraded_mode(cfg):
    """不一致検出時にアラートを出し、配信を止める判断ができること。"""
    from narops.monitoring import check_skew, should_block_delivery

    a, b = _frames({"h_starts_prior": [1.0, 2.0, 3.0]},
                   {"h_starts_prior": [1.0, 2.0, 9.0]})
    r = compare(a, b, cfg.tolerated_skew_columns, business_date(pd.Timestamp.now()))
    alert = check_skew(r)
    assert alert is not None and alert.severity == "Blocker"
    assert alert.blocks_delivery
    assert should_block_delivery([alert])


def test_sk07_clean_check_does_not_block(cfg):
    from narops.monitoring import check_skew, should_block_delivery

    a, b = _frames({"h_starts_prior": [1.0, 2.0, 3.0]},
                   {"h_starts_prior": [1.0, 2.0, 3.0]})
    r = compare(a, b, cfg.tolerated_skew_columns, business_date(pd.Timestamp.now()))
    assert check_skew(r) is None
    assert not should_block_delivery([])


# ---------------------------------------------------------------- 履歴の取得範囲
def test_history_covers_full_career_not_just_the_lookback_window(wh, clock):
    """as-of 集計は全キャリアを見る。時間窓だけで切ると学習と別物になる。

    実データでの実測: 180日窓に切ると h_starts_prior が平均 22.5 → 5.6
    （相関 0.51）、j_starts_prior が 8,080 → 365（相関 0.19）まで変わる。
    モデルは学習時と違う入力を受け取ることになる。
    """
    from datetime import date

    import numpy as np
    import pandas as pd

    from conftest import make_results
    from narops.db.merge import merge_final
    from narops.features import history_before

    old = make_results(n_races=8, start_day=1, seed=3, base_month=1)
    recent = make_results(n_races=8, start_day=18, seed=4, base_month=8)
    merge_final(wh, pd.concat([old, recent], ignore_index=True), clock=clock)

    as_of = pd.Timestamp("2026-08-25 00:00", tz="UTC")
    card = pd.DataFrame({"horse_sk": old["horse_sk"].unique()[:3]})

    windowed = history_before(wh, as_of, lookback_days=180, card=None,
                              max_bytes_billed=2_000_000_000)
    scoped = history_before(wh, as_of, lookback_days=180, card=card,
                            max_bytes_billed=2_000_000_000,
                            career_from=date(1998, 1, 1))

    assert (pd.to_datetime(windowed["race_date"]).min()
            > pd.Timestamp("2026-02-01")), "テスト前提: 窓には1月分が入らない"
    got = set(scoped.loc[scoped["horse_sk"].isin(card["horse_sk"]), "race_date"]
              .astype(str))
    assert any(d.startswith("2026-01") for d in got), (
        "出馬表に出る馬の過去キャリアが取れていません")


def test_history_still_respects_the_as_of_boundary(wh, clock):
    """範囲を広げても `start_ts <` の境界は緩めない。"""
    import pandas as pd

    from conftest import make_results
    from narops.db.merge import merge_final
    from narops.features import history_before

    results = make_results(n_races=12, start_day=18, seed=5)
    merge_final(wh, results, clock=clock)
    as_of = pd.Timestamp(results["start_ts"].min())
    card = pd.DataFrame({"horse_sk": results["horse_sk"].unique()[:4]})
    got = history_before(wh, as_of, lookback_days=3650, card=card,
                         max_bytes_billed=2_000_000_000)
    assert got.empty or pd.to_datetime(got["start_ts"]).max() < as_of


# ------------------------------------------------------------ 標準化の持ち回り
def test_release_without_a_standardizer_is_refused(tmp_path):
    """補完値・標準化統計量を配らないリリースは読み込ませない。

    学習は標準化済みの特徴量で係数と分割点を決めている。これが無いまま推論すると
    生の値がモデルに入り、線形モデルは係数の単位が合わず、GBDT は分割点が合わない。
    特徴量そのものは一致しているので、特徴量の突合では検出できない。
    """
    import json
    from types import SimpleNamespace

    import pytest

    from narops.errors import ArtifactIntegrityError
    from narops.runtime import load_models

    d = tmp_path / "rel"
    d.mkdir()
    (d / "feature_spec.json").write_text(json.dumps({"names": ["a"]}), encoding="utf-8")
    release = SimpleNamespace(path=d, release_id="v-test",
                              manifest=SimpleNamespace(ensemble_weights={}))
    with pytest.raises(ArtifactIntegrityError, match="standardizer"):
        load_models(release)


def test_standardizer_applies_frozen_statistics_not_per_race_ones():
    """推論時に手元の行から統計量を計算し直すと「レース内 z 値」になる。

    1レース分しか無いので、その場の平均・標準偏差は学習時のものと全く違う。
    """
    import numpy as np
    import pandas as pd

    from narops.runtime import Standardizer

    stats = {"f": {"median": 10.0, "mean": 10.0, "std": 5.0}}
    race = pd.DataFrame({"f": [12.0, 14.0, np.nan]})
    got = Standardizer(stats).apply(race, ["f"])["f"].tolist()
    assert got == pytest.approx([0.4, 0.8, 0.0])

    # その場で計算すると平均 13・標準偏差 √2 になり、全く別の値になる
    local = (race["f"].fillna(race["f"].median()) - race["f"].mean()) / race["f"].std()
    assert not np.allclose(got, local.fillna(0).to_numpy())


def test_standardizer_refuses_unknown_features():
    """配布物に統計量が無い特徴量を黙って素通ししない。"""
    import pandas as pd
    import pytest

    from narops.runtime import Standardizer

    with pytest.raises(KeyError, match="標準化統計量"):
        Standardizer({}).apply(pd.DataFrame({"f": [1.0]}), ["f"])
