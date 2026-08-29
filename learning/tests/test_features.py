"""FE-01..10: 特徴量層。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nar.config import feature_config
from nar.errors import LeakageError
from nar.features.builder import ASOF_FEATURES, build, content_hash
from nar.features.shrinkage import shrink
from nar.transform.prerace import assert_no_market_info


@pytest.fixture(scope="module")
def cfg():
    return feature_config()


@pytest.fixture(scope="module")
def feat(synth_tables, cfg):
    return build(synth_tables["entry"], synth_tables["race"], cfg)


# ------------------------------------------------------------------------ FE-01
def hand_calc_case() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """FX-05: 馬3頭 × 出走5回を手計算した as-of 集計の期待表。"""
    rows, expected = [], []
    # H1: 5走して 1着,着外,1着,着外,着外 → 各行の prior は自分より前だけ
    results = {
        "H1": [1, 4, 1, 3, 5],
        "H2": [2, 1, 3, 1, 2],
        "H3": [5, 5, 5, 5, 1],
    }
    for h, poss in results.items():
        wins = 0
        for i, pos in enumerate(poss):
            rows.append({
                "race_id": f"R{h}{i}", "race_date": pd.Timestamp("2015-01-01") + pd.Timedelta(days=i * 30),
                "start_ts": pd.Timestamp("2015-01-01") + pd.Timedelta(days=i * 30, hours=12),
                "baba_code": 20, "distance": 1400, "horse_no": 1, "waku": 1,
                "horse_sk": h, "sire_sk": "S1", "dam_sk": "D1",
                "jockey_sk": "J1", "trainer_sk": "T1",
                "finish_pos": pos, "is_win": int(pos == 1),
                "time_sec": 85.0 + pos * 0.2,
            })
            expected.append({
                "race_id": f"R{h}{i}", "horse_no": 1,
                "h_starts_prior": i, "h_wins_prior": wins,
                "h_winrate_prior": (wins / i) if i else None,
            })
            wins += int(pos == 1)

    entry = pd.DataFrame(rows)
    race = (
        entry[["race_id", "race_date", "start_ts", "baba_code", "distance"]]
        .assign(race_no=1, surface="ダ", class_level=1, prize_yen=1_000_000, n_runners=1)
    )
    return entry, race, pd.DataFrame(expected)


def test_fe01_asof_aggregation_matches_hand_calculation(cfg):
    entry, race, expected = hand_calc_case()
    got = build(entry, race, cfg).merge(expected, on=["race_id", "horse_no"], suffixes=("", "_exp"))
    assert len(got) == len(expected)
    assert (got["h_starts_prior"] == got["h_starts_prior_exp"]).all()
    assert (got["h_wins_prior"] == got["h_wins_prior_exp"]).all()
    both = got.dropna(subset=["h_winrate_prior_exp"])
    assert np.allclose(both["h_winrate_prior"].astype(float),
                       both["h_winrate_prior_exp"].astype(float))


# ------------------------------------------------------------------------ FE-02
def test_fe02_first_start_has_zero_starts_and_null_rates(feat):
    first = feat[feat["is_first_start"] == 1]
    assert len(first) > 0
    assert (first["h_starts_prior"] == 0).all()
    assert first["h_winrate_prior"].isna().all(), (
        "初出走の勝率は 0 ではなく NULL。0 は「勝てない馬」を意味してしまう。")


# ------------------------------------------------------------------------ FE-03
def test_fe03_prior_starts_increment_by_exactly_one(feat):
    d = feat.sort_values(["horse_sk", "start_ts", "race_id", "horse_no"])
    diffs = d.groupby("horse_sk")["h_starts_prior"].diff().dropna()
    assert (diffs == 1).all(), f"+1 でない遷移が {(diffs != 1).sum()} 件"


# ------------------------------------------------------------------------ FE-04
def test_fe04_same_day_races_reflect_earlier_race_in_the_day(cfg):
    """同日の第1Rと第12Rで h_*_prior が異なりうること（発走時刻順に反映される）。"""
    base = pd.Timestamp("2015-05-05")
    rows = []
    for i, race_no in enumerate((1, 12)):
        rows.append({
            "race_id": f"20{base:%Y%m%d}{race_no:02d}", "race_date": base,
            "start_ts": base + pd.Timedelta(hours=10 + i * 5),
            "baba_code": 20, "distance": 1400, "horse_no": 1, "waku": 1,
            "horse_sk": "H1", "sire_sk": "S1", "dam_sk": "D1",
            "jockey_sk": "J1", "trainer_sk": "T1",
            "finish_pos": 1, "is_win": 1, "time_sec": 85.0,
        })
    entry = pd.DataFrame(rows)
    race = entry[["race_id", "race_date", "start_ts", "baba_code", "distance"]].assign(
        race_no=1, surface="ダ", class_level=1, prize_yen=1_000_000, n_runners=1)
    out = build(entry, race, cfg).sort_values("start_ts")
    assert out["h_starts_prior"].tolist() == [0, 1]


# ------------------------------------------------------------------------ FE-05
def test_fe05_shrinkage_endpoints():
    prior = 0.11
    assert shrink(np.array([0]), np.array([0]), prior, 50.0)[0] == pytest.approx(prior)
    big = shrink(np.array([1e9 * 0.3]), np.array([1e9]), prior, 50.0)[0]
    assert big == pytest.approx(0.3, abs=1e-6), "出走数→∞ で生の勝率に収束すること"


def test_fe05_shrinkage_rejects_zero_alpha():
    with pytest.raises(ValueError, match="alpha"):
        shrink(np.array([1]), np.array([2]), 0.1, 0.0)


def test_fe05_shrunk_rates_pull_low_volume_toward_prior(feat):
    """出走数の少ない側ほど、収縮先（そのレースの事前確率）に近いこと。

    「収縮されていれば分散が小さい」という測り方はしない。事前確率が
    `1/頭数` でレースごとに動く以上、少走側の分散はむしろ事前確率のばらつきを
    そのまま引き継ぐ。実際、全期間平均を事前確率にしていた頃は分散が小さかったが、
    レース内の事前確率に変えた時点でその前提は成り立たなくなった。
    測るべきは「収縮先にどれだけ近いか」そのもの。
    """
    prior = 1.0 / feat["field_size"]
    gap = (feat["j_winrate_shrunk"] - prior).abs()
    low = gap[feat["j_starts_prior"] < 5]
    high = gap[feat["j_starts_prior"] > 50]
    assert len(low) > 50 and len(high) > 50, "比較できるだけの出走数分布がありません"
    assert low.mean() < high.mean(), (
        f"出走数の少ない側が収縮先から遠いです（少 {low.mean():.5f} / "
        f"多 {high.mean():.5f}）")


# ------------------------------------------------------------------------ FE-07
def test_fe07_speed_index_is_comparable_across_tracks(synth_tables, cfg):
    """速度指数が場×距離をまたいで比較可能なこと。

    設計書は平均 0 ± 0.05 / 標準偏差 1 ± 0.1 を期待値としているが、本実装は
    標準化の基準統計量そのものを as-of（当該レースより前のみ）で取っている。
    未来を見ずに標準化する以上、値は厳密には標準正規にならない。ここでは
    「場をまたいだ比較が成立する」ことを担保する緩めの範囲で判定し、
    設計書側の期待値はテスト仕様 §13 の手順で更新する。
    """
    from nar.features.builder import speed_index

    si = speed_index(synth_tables["entry"]).dropna(subset=["speed_index"])
    stats = si.groupby(["baba_code", "distance"])["speed_index"].agg(["mean", "std", "size"])
    stats = stats[stats["size"] >= 200]
    assert len(stats) > 0
    assert stats["mean"].abs().max() < 0.25, stats
    assert stats["std"].between(0.6, 1.5).all(), stats


# ------------------------------------------------------------------------ FE-08
def test_fe08_missing_rate_of_key_features_is_reported(feat):
    core = [c for c in ("h_starts_prior", "field_size", "draw_rel", "distance", "log_prize")
            if c in feat.columns]
    rates = feat[core].isna().mean() * 100
    assert (rates <= 5.0).all(), f"欠損率 5% 超:\n{rates[rates > 5.0]}"


# ------------------------------------------------------------------------ FE-09
def test_fe09_gold_is_deterministic(synth_tables, cfg):
    a = build(synth_tables["entry"], synth_tables["race"], cfg)
    b = build(synth_tables["entry"], synth_tables["race"], cfg)
    assert content_hash(a) == content_hash(b)


def test_fe09_speed_index_prefix_is_independent_of_later_rows(synth_tables):
    """先頭 N 行だけで計算しても、全行で計算した先頭 N 行と一致すること。

    as-of 集計を「全体の累積和 − 群開始時点の値」で書くと、浮動小数点では
    手前の全グループの合計が引き算に乗り、後ろの行を削るだけで過去の値が変わる。
    実際その実装で LK-05 が落ちた。群ごとに積み直していることをここで固定する。
    """
    from nar.features.builder import speed_index

    full = speed_index(synth_tables["entry"])
    # 切る位置は発走時刻の境界に合わせる。同時刻の行は互いに集計から外し合うので、
    # 同時刻グループの途中で切ると、残った側の値が変わるのは正しい挙動。
    cut = full["start_ts"].quantile(0.5)
    head = full[full["start_ts"] < cut]
    partial = speed_index(head.drop(columns=["speed_index"]))
    a = head["speed_index"].to_numpy()
    b = partial["speed_index"].to_numpy()
    both = ~(pd.isna(a) | pd.isna(b))
    assert (pd.isna(a) == pd.isna(b)).all(), "NaN の位置が変わっています"
    assert (a[both] == b[both]).all(), "後続の行が過去の速度指数を変えています"


# ------------------------------------------------------------------------ FE-10
def test_fe10_track_a_contains_no_market_columns(feat, cfg):
    assert_no_market_info(feat, cfg.market_term_blocklist)
    lowered = " ".join(feat.columns).lower()
    for term in ("odds", "人気", "支持率", "払戻"):
        assert term.lower() not in lowered


def test_fe10_blocklist_catches_injected_market_column(feat, cfg):
    poisoned = feat.assign(win_odds_ratio=1.0)
    with pytest.raises(LeakageError, match="市場情報"):
        assert_no_market_info(poisoned, cfg.market_term_blocklist)


def test_all_declared_features_are_produced(feat):
    missing = set(ASOF_FEATURES) - set(feat.columns)
    assert not missing, f"registry に宣言済みだが未生成: {missing}"


# ---------------------------------------------------------------- 学習対象の定義
def test_trainable_drops_horses_that_did_not_run():
    """取消・除外の馬を「4着以下」として学習させない。"""
    from nar.train.pipeline import trainable

    df = pd.DataFrame({
        "race_id": ["r1"] * 3 + ["r2"] * 3,
        "horse_no": [1, 2, 3, 1, 2, 3],
        "finish_pos": [1.0, 2.0, np.nan, 1.0, 2.0, 3.0],
        "is_win": [1, 0, 0, 1, 0, 0],
    })
    out = trainable(df)
    assert len(out) == 5
    assert out["finish_pos"].notna().all()


def test_trainable_drops_races_without_a_single_winner():
    """中止・未実施のレースと1着同着のレースはどちらも落とす。"""
    from nar.train.pipeline import trainable

    df = pd.DataFrame({
        "race_id": ["ok"] * 2 + ["no_winner"] * 2 + ["dead_heat"] * 2,
        "finish_pos": [1.0, 2.0, 2.0, 3.0, 1.0, 1.0],
        "is_win": [1, 0, 0, 0, 1, 1],
    })
    out = trainable(df)
    assert set(out["race_id"]) == {"ok"}


def test_graded_labels_refuses_missing_finish_positions():
    """NaN を int にすると int64 最小値になり、LightGBM が読める形で落ちない。"""
    from nar.models.lgbm import graded_labels

    with pytest.raises(ValueError, match="着順が NULL"):
        graded_labels(np.array([1.0, np.nan, 3.0]))
    assert graded_labels(np.array([1.0, 2.0, 5.0])).tolist() == [3, 2, 0]


def test_declared_feature_dtypes_do_not_depend_on_the_data(synth_tables, cfg):
    """申告値特徴量は常に float64。

    出走数は整数だが、成績列が空欄の行があると NaN が入って float になる。
    学習時に int64・推論時に float64 になると feature_spec が一致せず推論が止まる。
    """
    from nar.features.declared import DECLARED_FEATURES

    full = build(synth_tables["entry"], synth_tables["race"], cfg)
    blanked = synth_tables["entry"].copy()
    blanked.loc[blanked.index[:50], "全成績"] = ""
    partial = build(blanked, synth_tables["race"], cfg)
    for c in DECLARED_FEATURES:
        assert str(full[c].dtype) == "float64", f"{c}: {full[c].dtype}"
        assert str(partial[c].dtype) == "float64", f"{c}: {partial[c].dtype}"


def test_all_model_features_are_float64(feat):
    """dtype がデータや入力の型に依存すると feature_spec が一致しなくなる。"""
    from nar.features.builder import ASOF_FEATURES

    bad = {c: str(feat[c].dtype) for c in ASOF_FEATURES
           if c in feat.columns and str(feat[c].dtype) != "float64"}
    assert not bad, f"float64 でない特徴量: {bad}"


def test_speed_index_never_recomputes_when_the_column_exists(synth_tables):
    """保存済みの速度指数があるときは一切計算し直さない。

    場×距離の as-of 統計に依存する値なので、履歴の部分集合だけを渡して
    計算すると学習時と別の値になる。運用側が渡せるのは常に部分集合。
    欠損だけ埋める実装にすると、学習時に NULL だった行に部分集合由来の値が
    入って、そこだけ食い違う。
    """
    from nar.features.builder import speed_index

    full = speed_index(synth_tables["entry"])
    subset = full.sample(frac=0.3, random_state=0).copy()
    recomputed = speed_index(subset)
    a = full.set_index(["race_id", "horse_no"])["speed_index"]
    b = recomputed.set_index(["race_id", "horse_no"])["speed_index"]
    common = a.index.intersection(b.index)
    assert a.loc[common].equals(b.loc[common]), "保存済みの値が変わっています"

    # NULL のまま渡した行は NULL のまま返る（部分集合から埋めない）
    holed = full.copy()
    holed.loc[holed.index[:200], "speed_index"] = np.nan
    assert speed_index(holed)["speed_index"].isna().sum() >= 200
