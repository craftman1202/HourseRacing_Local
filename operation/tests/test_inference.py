"""IN-01..11: 推論エンドポイントの不変条件。"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from narops.clock import FixedClock, jst_datetime, to_utc
from narops.errors import (
    InsufficientData, NormalizationError, RaceExpired, ZeroFillForbidden,
)
from narops.inference import (
    PoolSizeModel, assert_no_zero_fill, assert_normalized, assert_within_window,
    blend, run_inference,
)
from narops.model.manifest import Manifest
from narops.shared import race_softmax

pytestmark = pytest.mark.unit

START = jst_datetime(2026, 8, 25, 20, 35)
N = 8


class LinearModel:
    def __init__(self, name: str, weights: np.ndarray) -> None:
        self.name = name
        self.w = weights

    def score(self, features: pd.DataFrame) -> np.ndarray:
        cols = ["f1", "f2"]
        return features[cols].to_numpy(dtype=float) @ self.w


@pytest.fixture
def features() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "horse_no": np.arange(1, N + 1),
        "f1": rng.normal(size=N), "f2": rng.normal(size=N),
    })


@pytest.fixture
def models() -> dict:
    return {"lgbm": LinearModel("lgbm", np.array([1.0, 0.5])),
            "tabm": LinearModel("tabm", np.array([0.8, 0.9]))}


@pytest.fixture
def manifests(release_dir) -> list[Manifest]:
    m = Manifest.read(release_dir / "manifest.json")
    return [m, m]


@pytest.fixture
def at_window() -> FixedClock:
    return FixedClock(START - timedelta(minutes=13))


@pytest.fixture
def odds() -> pd.Series:
    return pd.Series([3.2, 5.1, 8.0, 12.5, 4.4, 22.0, 60.0, 15.0])


# ------------------------------------------------------------------ IN-01
def test_in01_probabilities_sum_to_one(features, models, manifests, at_window, cfg, odds):
    res = run_inference(race_id="202026082511", features=features, models=models,
                        manifests=manifests, weights={"lgbm": 0.4, "tabm": 0.6},
                        temperature=1.0, clock=at_window, cfg=cfg, odds=odds)
    assert res.frame["p_win"].sum() == pytest.approx(1.0, abs=1e-9)
    assert (res.frame["p_win"] > 0).all()


def test_in01_broken_normalization_raises():
    with pytest.raises(NormalizationError, match="総和"):
        assert_normalized(np.array([0.5, 0.2]), np.array(["R1", "R1"]))


def test_in01_assertion_prevents_writing_results(features, models, manifests,
                                                 at_window, cfg, monkeypatch):
    """アサーション違反時は結果を書かない。届かないほうがまし。"""
    import narops.inference as inf

    monkeypatch.setattr(inf, "blend", lambda *a, **k: np.full(N, 0.5))
    with pytest.raises(NormalizationError):
        run_inference(race_id="R", features=features, models=models,
                      manifests=manifests, weights={"lgbm": 1.0}, temperature=0,
                      clock=at_window, cfg=cfg)


# ------------------------------------------------------------------ IN-02
def test_in02_scratched_horse_is_excluded_and_renormalized(
        features, models, manifests, at_window, cfg, odds):
    """出走取消の反映後に再正規化され、除外馬の候補が出ない。"""
    remaining = features.iloc[:-2].reset_index(drop=True)
    res = run_inference(race_id="R", features=remaining, models=models,
                        manifests=manifests, weights={"lgbm": 0.5, "tabm": 0.5},
                        temperature=1.0, clock=at_window, cfg=cfg,
                        odds=odds.iloc[:-2].reset_index(drop=True))
    assert len(res.frame) == N - 2
    assert res.frame["p_win"].sum() == pytest.approx(1.0, abs=1e-9)
    assert set(res.frame["horse_no"]) == set(range(1, N - 1))


# ------------------------------------------------------------------ IN-03
def test_in03_within_window_is_accepted(cfg):
    assert_within_window(START, FixedClock(START - timedelta(minutes=13)), cfg)
    assert_within_window(START, FixedClock(START - timedelta(minutes=13, seconds=45)), cfg)


def test_in03_after_start_is_expired(cfg):
    with pytest.raises(RaceExpired, match="発走時刻を過ぎ"):
        assert_within_window(START, FixedClock(START + timedelta(seconds=1)), cfg)


def test_in03_too_early_is_rejected(cfg):
    with pytest.raises(RaceExpired, match="早すぎ"):
        assert_within_window(START, FixedClock(START - timedelta(minutes=30)), cfg)


# ------------------------------------------------------------------ DR-05
def test_dr05_clock_skew_fails_conservatively(cfg):
    """5分ずらしたクロックで、締切超過の提案が出ない側に倒れること。"""
    late = FixedClock(START - timedelta(minutes=2), skew=timedelta(minutes=5))
    with pytest.raises(RaceExpired):
        assert_within_window(START, late, cfg)


# ------------------------------------------------------------------ IN-04
def test_in04_track_a_only_without_odds(features, models, manifests, at_window, cfg):
    res = run_inference(race_id="R", features=features, models=models,
                        manifests=manifests, weights={"lgbm": 0.5, "tabm": 0.5},
                        temperature=1.0, clock=at_window, cfg=cfg, odds=None)
    assert res.track_used == "A"
    assert res.frame["p_market"].isna().all()
    assert (res.frame["stake_yen"] == 0).all(), "オッズ無しでベット候補を出してはいけない"
    assert any("トラックA 単独" in n for n in res.notes)


def test_in04_track_ab_with_odds(features, models, manifests, at_window, cfg, odds):
    res = run_inference(race_id="R", features=features, models=models,
                        manifests=manifests, weights={"lgbm": 0.5, "tabm": 0.5},
                        temperature=1.0, clock=at_window, cfg=cfg, odds=odds)
    assert res.track_used == "A+B"
    assert res.frame["p_market"].notna().all()
    assert res.frame["p_market"].sum() == pytest.approx(1.0, abs=1e-9)


def test_in04_partial_odds_degrades_to_track_a(features, models, manifests,
                                               at_window, cfg, odds):
    partial = odds.copy()
    partial.iloc[3] = np.nan
    res = run_inference(race_id="R", features=features, models=models,
                        manifests=manifests, weights={"lgbm": 0.5, "tabm": 0.5},
                        temperature=1.0, clock=at_window, cfg=cfg, odds=partial)
    assert res.track_used == "A", "一部欠損で A+B を続けてはいけない"


# ------------------------------------------------------------------ IN-05
def test_in05_zero_fill_is_rejected():
    df = pd.DataFrame({"odds_win": [3.0, 0.0, 5.0]})
    with pytest.raises(ZeroFillForbidden, match="ゼロ埋め"):
        assert_no_zero_fill(df, ["odds_win"])


def test_in05_nan_odds_is_rejected():
    df = pd.DataFrame({"odds_win": [3.0, np.nan, 5.0]})
    with pytest.raises(ZeroFillForbidden, match="欠損"):
        assert_no_zero_fill(df, ["odds_win"])


def test_in05_zero_odds_input_raises_in_inference(features, models, manifests,
                                                  at_window, cfg, odds):
    bad = odds.copy()
    bad.iloc[0] = 0.0
    with pytest.raises(ZeroFillForbidden):
        run_inference(race_id="R", features=features, models=models,
                      manifests=manifests, weights={"lgbm": 1.0}, temperature=1.0,
                      clock=at_window, cfg=cfg, odds=bad)


# ------------------------------------------------------------------ IN-06
def test_in06_single_model_weight_reproduces_that_model(features, models,
                                                        manifests, at_window, cfg):
    from narops.shared import race_softmax

    rid = np.full(len(features), "R", dtype=object)
    solo = race_softmax(models["lgbm"].score(features), rid)
    res = run_inference(race_id="R", features=features, models=models,
                        manifests=manifests, weights={"lgbm": 1.0, "tabm": 0.0},
                        temperature=0, clock=at_window, cfg=cfg)
    assert np.allclose(res.frame["p_win"].to_numpy(), solo, atol=1e-9)


def test_in06_weights_must_sum_to_one(features, models):
    rid = np.full(len(features), "R", dtype=object)
    per = {"lgbm": np.full(len(features), 1 / len(features))}
    with pytest.raises(NormalizationError, match="総和1"):
        blend(per, {"lgbm": 0.7}, rid)


def test_in06_negative_weight_is_rejected(features, models):
    rid = np.full(len(features), "R", dtype=object)
    per = {"a": np.full(len(features), 1 / len(features)),
           "b": np.full(len(features), 1 / len(features))}
    with pytest.raises(NormalizationError):
        blend(per, {"a": 1.3, "b": -0.3}, rid)


# ------------------------------------------------------------------ IN-07
def test_in07_temperature_changes_probabilities_but_not_ranking(
        features, models, manifests, at_window, cfg):
    kw = dict(race_id="R", features=features, models=models, manifests=manifests,
              weights={"lgbm": 0.5, "tabm": 0.5}, clock=at_window, cfg=cfg)
    raw = run_inference(temperature=0, **kw).frame
    cal = run_inference(temperature=2.5, **kw).frame

    assert not np.allclose(raw["p_win"], cal["p_win"]), "較正が効いていません"
    assert (raw.sort_values("p_win", ascending=False)["horse_no"].tolist()
            == cal.sort_values("p_win", ascending=False)["horse_no"].tolist()), \
        "温度スケーリングは順位を変えてはいけません"


def test_in07_model_temperatures_calibrate_before_blending(
        features, models, manifests, at_window, cfg):
    """model_temperatures 指定時は、学習側（run_fold）と同じ「モデルごとに
    温度をかけてからアンサンブル」の順序になること（2026-09-17 修正、
    Design_LogicFlow.md §5-3）。

    合成後に単一温度をかける旧経路とは異なる結果になるはずで、かつ
    「各モデルを個別に較正してから blend() する」を素朴に手計算した値と
    一致することを確認する。
    """
    kw = dict(race_id="R", features=features, models=models, manifests=manifests,
              weights={"lgbm": 0.5, "tabm": 0.5}, clock=at_window, cfg=cfg)

    per_model_result = run_inference(
        temperature=1.0, model_temperatures={"lgbm": 0.7, "tabm": 1.8}, **kw).frame
    post_blend_result = run_inference(temperature=1.0, **kw).frame

    assert not np.allclose(per_model_result["p_win"], post_blend_result["p_win"]), \
        "モデルごとの較正は合成後の一律較正と同じ結果になってはいけません"

    # 手計算: 各モデルのレース内 softmax を個別に温度較正してから重み付き幾何平均
    rid = np.full(len(features), "R", dtype=object)
    per_model = {name: race_softmax(model.score(features), rid)
                for name, model in models.items()}
    calibrated = {
        "lgbm": race_softmax(np.log(np.clip(per_model["lgbm"], 1e-12, 1.0)) / 0.7, rid),
        "tabm": race_softmax(np.log(np.clip(per_model["tabm"], 1e-12, 1.0)) / 1.8, rid),
    }
    expected = blend(calibrated, {"lgbm": 0.5, "tabm": 0.5}, rid)
    assert np.allclose(per_model_result["p_win"].to_numpy(), expected, atol=1e-9)


# ------------------------------------------------------------------ IN-09
def test_in09_missing_horse_number_is_rejected(populated, release_dir):
    from nar.config import feature_config

    from narops.features import build_for_race

    card = pd.DataFrame({"race_id": ["R"] * 3, "horse_no": [1, None, 3],
                         "horse_sk": ["a", "b", "c"], "jockey_sk": ["j"] * 3,
                         "trainer_sk": ["t"] * 3, "sire_sk": ["s"] * 3})
    race_row = pd.Series({"race_id": "R", "race_date": START.date(),
                          "start_ts": to_utc(START), "baba_code": 20, "distance": 1200})
    with pytest.raises(InsufficientData, match="枠順"):
        build_for_race(populated, card, race_row,
                       Manifest.read(release_dir / "manifest.json"),
                       feature_config(), max_bytes_billed=10**9)


def test_in09_empty_card_is_rejected(populated, release_dir):
    from nar.config import feature_config

    from narops.features import build_for_race

    race_row = pd.Series({"race_id": "R", "race_date": START.date(),
                          "start_ts": to_utc(START), "baba_code": 20, "distance": 1200})
    with pytest.raises(InsufficientData, match="空"):
        build_for_race(populated, pd.DataFrame(columns=["horse_no"]), race_row,
                       Manifest.read(release_dir / "manifest.json"),
                       feature_config(), max_bytes_billed=10**9)


# ------------------------------------------------------------------ IN-11
def test_in11_retry_policy_distinguishes_transient_from_logic():
    from narops.errors import AsOfViolation, ZeroFillForbidden
    from narops.jobs import RetryPolicy

    p = RetryPolicy()
    assert p.should_retry(TimeoutError("bq"), attempt=1)
    assert p.should_retry(ConnectionError("net"), attempt=2)
    assert not p.should_retry(TimeoutError("bq"), attempt=3), "最大3回で打ち切る"
    # 同じ結果にしかならないものはリトライしない
    assert not p.should_retry(ZeroFillForbidden("x"), attempt=1)
    assert not p.should_retry(AsOfViolation("x"), attempt=1)


def test_in11_backoff_schedule():
    from narops.jobs import RetryPolicy

    p = RetryPolicy()
    assert [p.wait_for(i) for i in (1, 2, 3)] == [30, 90, 270]


# ------------------------------------------------------ 経済計算（自己インパクト）
def test_self_impact_correction_reduces_ev(features, models, manifests, at_window,
                                           cfg, odds):
    res = run_inference(race_id="R", features=features, models=models,
                        manifests=manifests, weights={"lgbm": 0.5, "tabm": 0.5},
                        temperature=1.0, clock=at_window, cfg=cfg, odds=odds,
                        pool_model=PoolSizeModel(default_yen=800_000))
    bet = res.frame[res.frame["stake_yen"] > 0]
    if len(bet):
        assert (bet["ev_adjusted"] <= bet["ev"] + 1e-12).all(), \
            "補正後 EV が補正前を上回っています"
        assert (bet["odds_effective"] <= bet["odds_win"] + 1e-12).all()


def test_pool_estimate_is_conservative():
    p = PoolSizeModel(default_yen=1_000_000, safety_factor=0.7)
    assert p.estimate(20, 1, 0) == pytest.approx(700_000), \
        "プールは小さめに見積もり、期待値を過大評価しない側に倒す"


def test_stake_never_exceeds_per_race_cap(features, models, manifests, at_window,
                                          cfg, odds):
    """上限は1頭ごとではなく**レース合計**に掛かること。

    単勝のみの現状では、Σp=1 かつ 1/4 Kelly なのでレース合計は
    0.25 × max_bet_per_race（¥750）が上界で、この不変条件は自然に満たされる。
    実際に効くのは Design_Operation.md §3.5 の同時ポートフォリオ最適化で
    1レースに複数券種が並んだときなので、**先に不変条件として固定しておく**。
    """
    res = run_inference(race_id="R", features=features, models=models,
                        manifests=manifests, weights={"lgbm": 0.5, "tabm": 0.5},
                        temperature=1.0, clock=at_window, cfg=cfg, odds=odds)
    assert (res.frame["stake_yen"] <= cfg.max_bet_per_race).all()
    assert res.frame["stake_yen"].sum() <= cfg.max_bet_per_race, (
        f"レース合計 {res.frame['stake_yen'].sum()} 円が1レース上限 "
        f"{cfg.max_bet_per_race} 円を超えています")
    assert (res.frame["stake_yen"] % 100 == 0).all(), "100円単位でない賭け額があります"


def test_per_race_cap_binds_when_several_horses_qualify(features, models, manifests,
                                                       at_window, cfg):
    """複数頭が同時に推奨される状況でもレース合計の上限を超えないこと。

    既定のフィクスチャは1頭しか賭け対象にならないので、複数頭が並ぶ経路を
    別に押さえる。単勝のみなら理論上界（¥750）のほうが先に効くため、この
    テストは上限式そのものの differentiator ではなく不変条件の確認である。
    差が出るのは複数券種を同時に建てるようになってから（§3.5）。
    """
    # 全頭の EV が閾値を大きく超えるオッズ。頭数倍の上限だと合計が
    # max_bet_per_race を超え、合計上限なら超えない。
    generous = pd.Series([30.0] * N)
    res = run_inference(race_id="R", features=features, models=models,
                        manifests=manifests, weights={"lgbm": 0.5, "tabm": 0.5},
                        temperature=1.0, clock=at_window, cfg=cfg, odds=generous)
    staked = (res.frame["stake_yen"] > 0).sum()
    assert staked >= 2, f"前提が崩れています（賭け対象 {staked} 頭）"
    assert res.frame["stake_yen"].sum() <= cfg.max_bet_per_race, (
        f"{staked} 頭で合計 {res.frame['stake_yen'].sum()} 円。"
        f"1レース上限 {cfg.max_bet_per_race} 円を超えています")


def test_day_budget_is_respected(features, models, manifests, at_window, cfg, odds):
    res = run_inference(race_id="R", features=features, models=models,
                        manifests=manifests, weights={"lgbm": 0.5, "tabm": 0.5},
                        temperature=1.0, clock=at_window, cfg=cfg, odds=odds,
                        day_budget_remaining=500)
    assert res.frame["stake_yen"].sum() <= 500


# --------------------------------------------------------- stake_hint_yen
def test_stake_hint_reflects_the_raw_kelly_amount_when_it_rounds_to_zero(
        features, models, manifests, at_window, cfg, odds):
    """Kelly 額が最低賭け金（100円）に届かない銘柄でも、丸め前の額が読めること。

    EV が基準を満たすのに `stake_yen` が0のとき、通知が空欄のままだと
    「なぜ推奨が無いのか」が運用側から読めない。単位未満で賭けないのと、
    EV 不足でそもそも賭ける理由が無いのを区別できるようにする。
    """
    # 賭け額を必ず最低単位未満に抑える設定（cap を極端に小さくする）
    cfg.kelly_fraction = 0.01
    res = run_inference(race_id="R", features=features, models=models,
                        manifests=manifests, weights={"lgbm": 0.5, "tabm": 0.5},
                        temperature=1.0, clock=at_window, cfg=cfg, odds=odds)
    passing = res.frame[res.frame["ev_adjusted"] >= cfg.discord_min_ev]
    assert len(passing), "この設定なら EV 基準を満たす行が残るはず（テスト前提）"
    assert (passing["stake_yen"] == 0).all(), "cap を絞ったのに実額が出ています（前提が崩れています）"
    assert (passing["stake_hint_yen"] > 0).all(), (
        "EV は基準を満たすのに参考額が0です。単位未満と検知不可を区別できていません")


def test_stake_hint_is_zero_once_ev_drops_below_threshold(
        features, models, manifests, at_window, cfg, odds):
    """EV 基準そのものを割った銘柄は、参考額も出さない（賭ける理由が無いため）。"""
    res = run_inference(race_id="R", features=features, models=models,
                        manifests=manifests, weights={"lgbm": 0.5, "tabm": 0.5},
                        temperature=1.0, clock=at_window, cfg=cfg, odds=odds)
    failing = res.frame[res.frame["ev_adjusted"] < cfg.discord_min_ev]
    if len(failing):
        assert (failing["stake_hint_yen"] == 0).all()


def test_stake_hint_is_consistent_with_the_rounded_stake(
        features, models, manifests, at_window, cfg, odds):
    """実額が出るときの参考額は、その丸め前の値そのものであること。

    `stake_hint_yen` は `stake_yen` を100円単位に丸める前の額なので、
    `floor(hint/100)*100 == stake_yen` が常に成り立つ（別々の計算をしていない）。
    """
    res = run_inference(race_id="R", features=features, models=models,
                        manifests=manifests, weights={"lgbm": 0.5, "tabm": 0.5},
                        temperature=1.0, clock=at_window, cfg=cfg, odds=odds,
                        pool_model=PoolSizeModel(default_yen=50_000_000))
    bet = res.frame[res.frame["stake_yen"] > 0]
    if len(bet):
        rounded = (bet["stake_hint_yen"] // 100) * 100
        assert (rounded == bet["stake_yen"]).all()
        assert (bet["stake_hint_yen"] >= bet["stake_yen"]).all()


def test_stake_hint_is_zero_without_odds(features, models, manifests, at_window, cfg):
    res = run_inference(race_id="R", features=features, models=models,
                        manifests=manifests, weights={"lgbm": 0.5, "tabm": 0.5},
                        temperature=1.0, clock=at_window, cfg=cfg, odds=None)
    assert (res.frame["stake_hint_yen"] == 0).all()


# --------------------------------------------------------------- p_top3
def test_run_inference_includes_a_place_probability(features, models, manifests,
                                                     at_window, cfg, odds):
    """単勝しか予測しないモデルから、複勝（上位3着以内）確率も出すこと。

    Discord 通知に単勝・複勝を並べて出すための入力（narops.discord.format）。
    Harville 式の実装そのものの正しさは学習側 tests/test_eval.py が検証する。
    """
    res = run_inference(race_id="R", features=features, models=models,
                        manifests=manifests, weights={"lgbm": 0.5, "tabm": 0.5},
                        temperature=1.0, clock=at_window, cfg=cfg, odds=odds)
    assert "p_top3" in res.frame.columns
    assert res.frame["p_top3"].sum() == pytest.approx(min(3, len(res.frame)))
    assert (res.frame["p_top3"] >= res.frame["p_win"] - 1e-9).all(), (
        "複勝確率が単勝確率を下回っています（上位3着は1着を含むはず）")
