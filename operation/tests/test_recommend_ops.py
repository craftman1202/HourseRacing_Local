"""全券種の推奨買い目の運用側（2026-10-13）。

組み方の単一実装は学習側 nar.eval.recommend（learning/tests/test_recommend.py）。
ここでは ops.yaml の規則・推論・記録（参考は記録しない）・通知を固定する。
"""

from __future__ import annotations

from dataclasses import replace
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
from narops.shared import BetPlan, Rule, popularity_from_odds, recommend_structures

pytestmark = pytest.mark.unit
START = jst_datetime(2026, 10, 14, 15, 30)
ODDS6 = [3.0, 1.4, 6.0, 9.0, 15.0, 30.0]          # 2番が1番人気
WAKU6 = [1, 2, 3, 4, 5, 6]


def _dev():
    return DeviationModel(features=["f1"], booster=None, ridge_coef=np.array([0.0]),
                          ridge_intercept=0.0,
                          meta={"pop_table": {}, "reg_weights": {"ridge": 1.0},
                                "reg_calibration": {"ridge": {"gamma": 0.1, "temperature": 1.0}}})


def _infer(release_dir, cfg, odds=ODDS6, waku=WAKU6, budget=50_000):
    n = len(odds)
    m = Manifest.read(Path(release_dir) / "manifest.json")
    feats = pd.DataFrame({"horse_no": np.arange(1, n + 1), "f1": np.zeros(n)})
    odds = pd.Series(odds)
    return run_inference(race_id="202026101401", features=feats, models={}, manifests=[m],
                         weights={}, temperature=1.0,
                         clock=FixedClock(START - timedelta(minutes=10)), cfg=cfg, odds=odds,
                         deviation=_dev(), popularity=popularity_from_odds(odds).to_numpy(),
                         waku=None if waku is None else np.asarray(waku, dtype=float),
                         day_budget_remaining=budget)


def test_ops_yaml_rules_are_known_structures_and_reference_only():
    c = OpsConfig.load()
    known = recommend_structures()
    assert {r.structure for r in c.recommend} <= set(known)
    bets = {known[r.structure][0] for r in c.recommend}
    assert {"wide", "quinella", "exacta", "trio", "trifecta", "bracket_q", "bracket_e"} <= bets
    assert not any(r.stake for r in c.recommend)
    assert c.recommend_pop_table and "1_6" in c.recommend_pop_table


def test_every_rule_fires_with_a_strong_favourite(release_dir, cfg):
    # 勝率表で 6 頭立ての1番人気 ≈ 0.50。λ=0 の規則はモデル側の確率で判定するので、
    # ここでは全規則を λ=1・min_p=0 にして「全券種が組める」ことだけを見る
    rules = tuple(replace(r, lam=1.0, min_p=0.0) for r in cfg.recommend)
    res = _infer(release_dir, replace(cfg, recommend=rules))
    got = {pl.bet_type for pl in res.exotic}
    assert got == {"ワイド", "馬連", "馬単", "3連複", "3連単", "枠連", "枠単"}
    by = {pl.bet_type: pl for pl in res.exotic}
    assert by["馬連"].tickets == [(1, 2)]                 # 人気1位=2番、2位=1番
    assert by["馬単"].tickets == [(2, 1), (1, 2)]
    assert all(pl.axis == 2 for pl in res.exotic)


def test_bracket_rules_need_waku(release_dir, cfg):
    rules = tuple(replace(r, lam=1.0, min_p=0.0) for r in cfg.recommend)
    res = _infer(release_dir, replace(cfg, recommend=rules), waku=None)
    assert not {"枠連", "枠単"} & {pl.bet_type for pl in res.exotic}


def test_reference_plans_are_not_recorded(wh, clock):
    ref = BetPlan("馬連", "馬連 1点", 2, [1], 0.55, [(1, 2)], 100, stake=False)
    staked = BetPlan("3連単", "3連単 1・2着固定 3着2頭", 2, [1, 3, 4], 0.55,
                     [(2, 1, 3), (2, 1, 4)], 100, stake=True)
    frame = pd.DataFrame({"horse_no": [2], "stake_yen": [0]})
    n = write_bet_candidates(wh, "202026101401", frame, "v", date(2026, 10, 14), clock,
                             exotic=[ref, staked])
    assert n == 1
    got = wh.query("SELECT horse_no, bet_type, stake_yen FROM bet_candidate WHERE race_date = ?",
                   [date(2026, 10, 14)])
    assert got.to_dict("records") == [{"horse_no": 2, "bet_type": "3連単", "stake_yen": 200}]


def _cand():
    return pd.DataFrame({"horse_no": [1, 2], "p_win": [0.2, 0.5], "p_top3": [0.6, 0.9],
                         "ev_adjusted": [0.8, 0.9], "ev_place": [0.8, 0.9],
                         "stake_yen": [0, 0], "bet_type": [None, None]})


def _embed(plans):
    return race_embed(race_id="R", track_name="笠松", race_no=3, class_name="C1", distance=1400,
                      start_ts=to_utc(START), now=to_utc(START) - timedelta(minutes=10),
                      model_release="v", track_used="A+B", candidates=_cand(), min_ev=1.05,
                      horse_names={2: "ホシノヒカリ"}, exotic=plans, day_budget_remaining=50_000)


def test_notification_lists_reference_plans_without_counting_them():
    plans = [BetPlan("馬連", "馬連 1点", 2, [1], 0.55, [(1, 2)], 100, note="検証ROI 0.83"),
             BetPlan("3連単", "3連単 1・2着固定 3着2頭", 2, [1, 3, 4], 0.55,
                     [(2, 1, 3), (2, 1, 4)], 100)]
    d = _embed(plans).description
    assert "参考" in d and "馬連 1点" in d and "1-2" in d and "検証ROI 0.83" in d
    assert "2→1→3  2→1→4" in d and "2点 × ¥100 = **¥200**" in d
    assert "買い目合計 —" in d                        # 参考は合計に入れない（0 円は — 表示）


def test_bracket_only_does_not_trigger_a_notification():
    pl = BetPlan("枠連", "枠連 1点", 2, [1], 0.35, [(1, 2)], 100, standalone=False)
    assert _embed([pl]) is None
    other = BetPlan("ワイド", "ワイド 1点", 2, [1], 0.55, [(1, 2)], 100)
    d = _embed([other, pl]).description
    assert "枠連 1点" in d and "ワイド 1点" in d


def test_rule_dataclass_defaults():
    r = Rule("ワイド 1点", 0.0, 0.5)
    assert r.stake is False and r.standalone is True and r.unit_yen == 100
