"""3連単「1番人気1着固定流し」の運用側（2026-10-06）。

判定の単一実装は学習側 nar.eval.exotic（learning/tests/test_exotic.py）。
ここでは推論・記録・通知・1日の残枠の扱いを固定する。
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from narops.clock import FixedClock, jst_datetime, to_utc
from narops.config import OpsConfig
from narops.discord.format import race_embed
from narops.inference import run_inference
from narops.model.manifest import Manifest
from narops.pipeline import write_bet_candidates
from narops.runtime import DeviationModel
from narops.shared import TrifectaPlan, popularity_from_odds

pytestmark = pytest.mark.unit
START = jst_datetime(2026, 10, 7, 15, 30)
# 6頭立て。1番人気 0.55 → 条件（≥0.5）を満たす
TABLE6 = {"1_6": 0.55, "2_6": 0.18, "3_6": 0.11, "4_6": 0.08, "5_6": 0.05, "6_6": 0.03}


def _dev(table):
    return DeviationModel(features=["f1"], booster=None, ridge_coef=np.array([0.0]),
                          ridge_intercept=0.0,
                          meta={"pop_table": table, "reg_weights": {"ridge": 1.0},
                                "reg_calibration": {"ridge": {"gamma": 0.1, "temperature": 1.0}}})


def _infer(release_dir, cfg, table, odds, budget=50_000):
    n = len(odds)
    m = Manifest.read(Path(release_dir) / "manifest.json")
    feats = pd.DataFrame({"horse_no": np.arange(1, n + 1), "f1": np.zeros(n)})
    odds = pd.Series(odds)
    return run_inference(race_id="202026100701", features=feats, models={}, manifests=[m],
                         weights={}, temperature=1.0,
                         clock=FixedClock(START - timedelta(minutes=10)), cfg=cfg, odds=odds,
                         deviation=_dev(table), popularity=popularity_from_odds(odds).to_numpy(),
                         day_budget_remaining=budget)


def test_ops_yaml_enables_the_validated_rule():
    t = OpsConfig.load().trifecta
    assert t and (t["min_p"], t["n_partners"], t["unit_yen"]) == (0.5, 8, 100)


def test_small_field_with_a_strong_favourite_buys_the_full_spread(release_dir, cfg):
    res = _infer(release_dir, cfg, TABLE6, [3.0, 1.4, 6.0, 9.0, 15.0, 30.0])
    [plan] = res.exotic
    assert plan.axis == 2                       # 1番人気（オッズ 1.4）
    assert plan.partners == [1, 3, 4, 5, 6]     # 人気順
    assert len(plan.tickets) == 20 and plan.stake_yen == 2000
    assert all(t[0] == 2 for t in plan.tickets)


def test_no_bet_when_the_favourite_is_below_the_bar(release_dir, cfg):
    weak = {**TABLE6, "1_6": 0.40}
    assert _infer(release_dir, cfg, weak, [3.0, 1.4, 6.0, 9.0, 15.0, 30.0]).exotic == []


def test_no_bet_when_popularity_is_missing(release_dir, cfg):
    res = _infer(release_dir, cfg, TABLE6, [3.0, np.nan, 6.0, 9.0, 15.0, 30.0])
    assert res.exotic == []


def test_skipped_entirely_when_the_day_budget_cannot_cover_all_tickets(release_dir, cfg):
    res = _infer(release_dir, cfg, TABLE6, [3.0, 1.4, 6.0, 9.0, 15.0, 30.0], budget=1500)
    assert res.exotic == []
    assert any("見送り" in n for n in res.notes)


def test_bet_candidate_records_one_row_with_the_total(wh, clock):
    plan = TrifectaPlan(axis=2, partners=[1, 3], p_axis=0.55,
                        tickets=[(2, 1, 3), (2, 3, 1)], unit_yen=100)
    frame = pd.DataFrame({"horse_no": [2], "stake_yen": [0]})
    n = write_bet_candidates(wh, "202026100701", frame, "v", date(2026, 10, 7), clock,
                             exotic=[plan])
    assert n == 1
    got = wh.query("SELECT horse_no, bet_type, stake_yen FROM bet_candidate WHERE race_date = ?",
                   [date(2026, 10, 7)])
    assert got.to_dict("records") == [{"horse_no": 2, "bet_type": "3連単", "stake_yen": 200}]


def test_notification_shows_the_trifecta_even_without_win_place_bets():
    plan = TrifectaPlan(axis=2, partners=[1, 3, 4, 5, 6], p_axis=0.55,
                        tickets=[(2, b, c) for b in [1, 3, 4, 5, 6] for c in [1, 3, 4, 5, 6]
                                 if b != c], unit_yen=100)
    cand = pd.DataFrame({"horse_no": [1, 2], "p_win": [0.2, 0.5], "p_top3": [0.6, 0.9],
                         "ev_adjusted": [0.8, 0.9], "ev_place": [0.8, 0.9],
                         "stake_yen": [0, 0], "bet_type": [None, None]})
    e = race_embed(race_id="R", track_name="笠松", race_no=3, class_name="C1", distance=1400,
                   start_ts=to_utc(START), now=to_utc(START) - timedelta(minutes=10),
                   model_release="v", track_used="A+B", candidates=cand, min_ev=1.05,
                   horse_names={2: "ホシノヒカリ"}, exotic=[plan])
    assert e is not None
    d = e.description
    assert "3連単 1着固定流し" in d and "1着 2番 ホシノヒカリ" in d
    assert "2・3着 1,3,4,5,6" in d and "20点 × ¥100 = **¥2,000**" in d
    assert "買い目合計 ¥2,000" in d and "```" not in d


def test_no_exotic_and_no_candidates_means_no_notification():
    cand = pd.DataFrame({"horse_no": [1], "p_win": [0.2], "p_top3": [0.6], "ev_adjusted": [0.8],
                         "ev_place": [0.8], "stake_yen": [0], "bet_type": [None]})
    assert race_embed(race_id="R", track_name="笠松", race_no=3, class_name="C1", distance=1400,
                      start_ts=to_utc(START), now=to_utc(START) - timedelta(minutes=10),
                      model_release="v", track_used="A+B", candidates=cand, min_ev=1.05,
                      exotic=[]) is None
