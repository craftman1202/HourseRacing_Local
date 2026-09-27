"""複勝モデルと「単勝・複勝の EV の高い方を1点」（2026-09-27）の運用側。

判定・金額の単一実装は学習側 `nar.eval.place`（learning/tests/test_place.py）。
ここでは運用側の配線 — オッズ解析・推論・記録・通知・投票テキスト — を固定する。
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from narops.api import HorsePrediction, Principal, RacePrediction, build_app
from narops.clock import FixedClock, jst_datetime, to_utc
from narops.config import OpsConfig, StrategyConfig
from narops.discord.format import race_embed
from narops.inference import run_inference
from narops.model.manifest import Manifest
from narops.nar_source import parse_win_odds
from narops.pipeline import write_bet_candidates
from narops.runtime import PlaceModels

pytestmark = pytest.mark.unit

START = jst_datetime(2026, 9, 28, 15, 30)
N = 8

# OddsTanFuku の実際の見出し（2026-09-26 高知 5R で確認）
TANFUKU_HTML = """<table>
  <tr><th>枠</th><th>馬番</th><th>馬名</th><th>単勝 オッズ</th>
      <th>複勝オッズ (3着払い)</th><th>複勝オッズ (3着払い)</th></tr>
  <tr><td>1</td><td>1</td><td>ウォーターレモン</td><td>110.2</td><td>6.2-</td><td>12.8</td></tr>
  <tr><td>2</td><td>2</td><td>ベストディシジョン</td><td>4.8</td><td>1.2-</td><td>2.1</td></tr>
  <tr><td>3</td><td>3</td><td>取消馬</td><td>1.5</td><td>-</td><td>-</td></tr>
</table>"""


class Linear:
    def __init__(self, name, w):
        self.name, self.w = name, np.asarray(w, dtype=float)

    def score(self, features):
        return features[["f1", "f2"]].to_numpy(dtype=float) @ self.w


@pytest.fixture
def features():
    # 馬1が突出して強い（単勝・複勝とも確率が高い）
    return pd.DataFrame({"horse_no": np.arange(1, N + 1),
                         "f1": [3.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0],
                         "f2": np.zeros(N)})


@pytest.fixture
def place_models():
    return PlaceModels(models={"lgbm": Linear("place_lgbm", [1.0, 0.0]),
                               "tabm": Linear("place_tabm", [0.9, 0.0])},
                       temperatures={"lgbm": 1.0, "tabm": 1.0},
                       weights={"lgbm": 0.7, "tabm": 0.3}, ensemble_temperature=1.0,
                       release_meta={})


@pytest.fixture
def strategy(cfg) -> StrategyConfig:
    assert cfg.strategy is not None, "conf/ops.yaml に strategy: max_ev がありません"
    return cfg.strategy


def _infer(features, release_dir, cfg, place, strategy, odds, place_odds, budget=None):
    m = Manifest.read(release_dir / "manifest.json")
    return run_inference(
        race_id="202026092801", features=features,
        models={"lgbm": Linear("lgbm", [1.0, 0.0]), "tabm": Linear("tabm", [0.9, 0.0])},
        manifests=[m, m], weights={"lgbm": 0.5, "tabm": 0.5}, temperature=1.0,
        clock=FixedClock(START - timedelta(minutes=10)), cfg=cfg, odds=odds,
        place=place, place_odds=place_odds, strategy=strategy,
        day_budget_remaining=budget)


# ------------------------------------------------------------------- 設定
def test_ops_yaml_carries_the_backtested_strategy_parameters():
    s = OpsConfig.load().strategy
    assert (s.min_ev, s.min_prob, s.kelly_scale) == (1.0, 0.6, 1.0)
    assert (s.budget_win_per_race, s.budget_place_per_race) == (10_000, 10_000)
    assert s.place_odds_alpha == pytest.approx(0.257)


# ------------------------------------------------------------------ オッズ
def test_parse_reads_place_odds_range_next_to_win_odds():
    out = parse_win_odds(TANFUKU_HTML, "322026092605")
    assert list(out["odds_win"]) == [110.2, 4.8, 1.5]
    assert list(out["pl_min"][:2]) == [6.2, 1.2]
    assert list(out["pl_max"][:2]) == [12.8, 2.1]
    assert out["pl_min"].isna().iloc[2], "複勝オッズが読めない馬は NaN（判定に使わない）"


# ------------------------------------------------------------------- 推論
def test_place_probability_comes_from_the_place_model_and_sums_to_slots(
        features, release_dir, cfg, place_models, strategy):
    odds = pd.Series([1.8, 4.0, 6.0, 9.0, 15.0, 25.0, 40.0, 80.0])
    pl = pd.DataFrame({"pl_min": [1.1] * N, "pl_max": [1.3] * N})
    res = _infer(features, release_dir, cfg, place_models, strategy, odds, pl)
    assert res.frame["p_place"].sum() == pytest.approx(3.0)   # 8頭立て = 3着払い
    assert (res.frame["p_top3"] == res.frame["p_place"]).all()
    assert {"ev_place", "bet_type", "ev_bet"} <= set(res.frame.columns)


def test_bets_follow_max_ev_rule_with_kelly_amounts(features, release_dir, cfg,
                                                   place_models, strategy):
    odds = pd.Series([1.8, 4.0, 6.0, 9.0, 15.0, 25.0, 40.0, 80.0])
    pl = pd.DataFrame({"pl_min": [1.2] * N, "pl_max": [1.6] * N})
    res = _infer(features, release_dir, cfg, place_models, strategy, odds, pl)
    f = res.frame
    top = f.iloc[0]
    assert top["stake_yen"] > 0
    ev_w, ev_p = top["p_win"] * 1.8, top["p_place"] * (1.2 + 0.257 * 0.4)
    assert top["bet_type"] == ("単勝" if ev_w >= ev_p else "複勝")
    assert top["ev_bet"] == pytest.approx(max(ev_w, ev_p))
    # 賭けない馬は券種も EV も持たない
    assert f.loc[f["stake_yen"] == 0, "bet_type"].isna().all()
    assert f["stake_yen"].max() <= 10_000 and (f["stake_yen"] % 100 == 0).all()
    assert len(res.bet_candidates(cfg.discord_min_ev)) == int((f["stake_yen"] > 0).sum())


def test_day_budget_still_caps_the_total(features, release_dir, cfg, place_models, strategy):
    odds = pd.Series([1.8, 4.0, 6.0, 9.0, 15.0, 25.0, 40.0, 80.0])
    pl = pd.DataFrame({"pl_min": [1.2] * N, "pl_max": [1.6] * N})
    res = _infer(features, release_dir, cfg, place_models, strategy, odds, pl, budget=300)
    assert res.frame["stake_yen"].sum() <= 300


def test_without_place_odds_only_win_is_considered(features, release_dir, cfg,
                                                   place_models, strategy):
    odds = pd.Series([1.8, 4.0, 6.0, 9.0, 15.0, 25.0, 40.0, 80.0])
    res = _infer(features, release_dir, cfg, place_models, strategy, odds, None)
    assert res.frame["ev_place"].isna().all()
    assert set(res.frame["bet_type"].dropna()) <= {"単勝"}
    assert any("複勝オッズ未取得" in n for n in res.notes)


def test_without_a_place_model_the_legacy_win_only_path_is_kept(features, release_dir, cfg,
                                                                strategy):
    """ばんえい・旧リリース: 単勝モデルの Harville で複勝の賭け金を決めない。"""
    odds = pd.Series([1.8, 4.0, 6.0, 9.0, 15.0, 25.0, 40.0, 80.0])
    res = _infer(features, release_dir, cfg, None, strategy, odds, None)
    assert "ev_place" not in res.frame.columns
    assert "bet_type" not in res.frame.columns


# ------------------------------------------------------------------- 記録
def test_bet_candidate_rows_carry_each_horses_bet_type(wh, clock):
    frame = pd.DataFrame({"horse_no": [1, 2, 3], "stake_yen": [2100, 800, 0],
                          "bet_type": ["単勝", "複勝", None], "ev": [1.17, 0.6, 0.5],
                          "ev_adjusted": [1.17, 0.6, 0.5], "ev_bet": [1.17, 1.12, np.nan],
                          "kelly": [0.21, 0.08, 0.0]})
    n = write_bet_candidates(wh, "202026092801", frame, "v-test", date(2026, 9, 28), clock)
    assert n == 2
    got = wh.query("SELECT horse_no, bet_type, ev FROM bet_candidate WHERE race_date = ? "
                   "ORDER BY horse_no", [date(2026, 9, 28)])
    assert list(got["bet_type"]) == ["単勝", "複勝"]
    assert got["ev"].iloc[1] == pytest.approx(1.12), "複勝を買う馬は複勝の EV を記録する"


# ------------------------------------------------------------------- 通知
def _cand():
    return pd.DataFrame({
        "horse_no": [1, 2, 3], "p_win": [0.62, 0.25, 0.05], "p_top3": [0.92, 0.70, 0.20],
        "ev_adjusted": [1.12, 0.90, 1.40], "ev_place": [1.05, 1.15, 0.30],
        "stake_yen": [2100, 1200, 0], "bet_type": ["単勝", "複勝", None],
    })


def test_notification_shows_both_evs_and_the_bet_type():
    e = race_embed(race_id="R", track_name="高知", race_no=5, class_name="C3", distance=1300,
                   start_ts=to_utc(START), now=to_utc(START) - timedelta(minutes=10),
                   model_release="v2026.09.27-A", track_used="A+B", candidates=_cand(),
                   min_ev=1.05)
    assert e is not None
    d = e.description
    assert "単EV" in d and "複EV" in d
    assert "1.12" in d and "1.05" in d and "1.15" in d
    assert "単¥2,100" in d and "複¥1,200" in d
    assert "買い目合計 ¥3,300" in d


def test_notification_lists_high_ev_horses_even_without_a_bet():
    """単勝 EV 1.40 の馬3は確率が足りず賭けないが、EV は通知に載る。"""
    e = race_embed(race_id="R", track_name="高知", race_no=5, class_name="C3", distance=1300,
                   start_ts=to_utc(START), now=to_utc(START) - timedelta(minutes=10),
                   model_release="v", track_used="A+B", candidates=_cand(), min_ev=1.05)
    assert "   3  " in e.description and "1.40" in e.description


def test_notification_is_skipped_when_nothing_clears_the_bar():
    c = _cand().assign(ev_adjusted=[0.9, 0.8, 0.7], ev_place=[0.9, 0.8, 0.7], stake_yen=0,
                       bet_type=None)
    e = race_embed(race_id="R", track_name="高知", race_no=5, class_name="C3", distance=1300,
                   start_ts=to_utc(START), now=to_utc(START) - timedelta(minutes=10),
                   model_release="v", track_used="A+B", candidates=c, min_ev=1.05)
    assert e is None


# ----------------------------------------------------------------- 投票テキスト
def test_betslip_uses_each_horses_bet_type():
    from fastapi.testclient import TestClient

    pred = RacePrediction(
        race_id="322026092805", race_date=date(2026, 9, 28), start_ts=to_utc(START),
        track_name="高知", race_no=5, model_release="v", track_used="A+B", status="ok",
        horses=[HorsePrediction(horse_no=1, p_win=0.7, stake_yen=2100, bet_type="単勝"),
                HorsePrediction(horse_no=2, p_win=0.3, stake_yen=1200, bet_type="複勝")])
    app = build_app({"race": lambda rid: pred, "principal": Principal("u", "paid")})
    text = TestClient(app).get("/races/322026092805/betslip?site=rakuten").json()["text"]
    assert text.splitlines() == ["32,20260928,5,T,1,21", "32,20260928,5,F,2,12"]
