"""複勝確率と「単勝・複勝の EV の高い方を1点」（nar.eval.place）。"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from nar.eval import metrics
from nar.eval.place import (
    estimated_place_odds, harville_topk, kelly_stakes, max_ev_bets,
    max_ev_bets_with_fallback, min_ev_bets, place_probability, place_slots,
)


def test_place_slots_follow_the_nar_rule():
    assert list(place_slots(np.array([3, 4, 5, 7, 8, 16]))) == [0, 0, 2, 2, 3, 3]


@pytest.mark.parametrize("n", [5, 7, 8, 12])
def test_harville_topk_matches_the_enumerating_reference(n):
    rng = np.random.default_rng(n)
    p = rng.random(n)
    p /= p.sum()
    k = int(place_slots(n))
    rid = np.array(["R"] * n)
    got = harville_topk(p, rid, np.array([k]))
    ref = metrics.harville_place_probability(p, rid, k=k)
    assert np.abs(got - ref).max() < 1e-12
    assert got.sum() == pytest.approx(k)


def test_harville_topk_brute_force_three_places():
    rng = np.random.default_rng(3)
    p = rng.random(6)
    p /= p.sum()
    brute = np.zeros(6)
    for order in itertools.permutations(range(6), 3):
        pr, left = 1.0, 1.0
        for i in order:
            pr *= p[i] / left
            left -= p[i]
        for i in order:
            brute[i] += pr
    got = harville_topk(p, np.array(["R"] * 6), np.array([3]))
    assert np.abs(got - brute).max() < 1e-12


def test_harville_topk_handles_races_with_different_slots():
    rid = np.array(["A"] * 8 + ["B"] * 6)
    p = np.r_[np.full(8, 1 / 8), np.full(6, 1 / 6)]
    got = harville_topk(p, rid, np.array([3] * 8 + [2] * 6))
    assert got[:8].sum() == pytest.approx(3)
    assert got[8:].sum() == pytest.approx(2)


def test_place_probability_sums_to_slots_and_ignores_zero_weight_models():
    rid = np.array(["R"] * 9)
    scores = {"lgbm": np.linspace(-1, 1, 9), "tabm": np.linspace(1, -1, 9)}
    p = place_probability(scores, {"lgbm": 1.1, "tabm": 0.8},
                          {"clogit": 0.0, "lgbm": 0.7, "tabm": 0.3}, 1.0, rid,
                          np.full(9, 3))
    assert p.sum() == pytest.approx(3)
    assert (p > 0).all() and (p < 1).all()


def test_estimated_place_odds_interpolates_the_range():
    assert estimated_place_odds([1.2], [2.0], 0.25)[0] == pytest.approx(1.4)


def test_kelly_stakes_cap_the_race_total_at_the_budget():
    rid = np.array(["R", "R", "R"])
    stake, f = kelly_stakes(np.array([0.9, 0.9, 0.9]), np.array([3.0, 3.0, 3.0]),
                            np.array([True, True, True]), rid, budget=10_000)
    assert f.sum() == pytest.approx(1.0)
    assert stake.sum() <= 10_000
    assert (stake % 100 == 0).all()


def test_max_ev_bets_picks_the_higher_ev_side_and_applies_both_thresholds():
    rid = np.array(["R"] * 4)
    out = max_ev_bets(
        rid,
        p_win=np.array([0.65, 0.30, 0.62, 0.05]),
        odds_win=np.array([1.8, 2.0, 1.5, 30.0]),
        p_place=np.array([0.90, 0.70, 0.95, 0.30]),
        place_odds=np.array([1.1, 1.6, 1.2, 5.0]),
        min_ev=1.0, min_prob=0.6)
    # 0: 単 EV 1.17 > 複 0.99 → 単勝 / 1: 単 0.6 < 複 1.12 → 複勝（p_place 0.7 ≥ 0.6）
    # 2: 単 0.93 < 複 1.14 → 複勝 / 3: 複 EV 1.5 だが p_place 0.3 < 0.6 → 見送り
    assert list(out["bet_type"]) == ["単勝", "複勝", "複勝", None]
    assert (out["stake_yen"].iloc[:3] > 0).all() and out["stake_yen"].iloc[3] == 0
    # 単勝のケリー: (0.65×1.8−1)/(0.8) = 0.2125 → ¥2,100
    assert out["stake_yen"].iloc[0] == 2100


def test_max_ev_bets_falls_back_to_win_when_place_odds_are_missing():
    out = max_ev_bets(np.array(["R", "R"]), np.array([0.7, 0.3]), np.array([1.6, 3.0]),
                      np.array([0.9, 0.6]), np.array([np.nan, np.nan]))
    assert out["bet_type"].iloc[0] == "単勝"
    assert np.isnan(out["ev_place"]).all()


def test_min_ev_bets_picks_the_lower_ev_side():
    rid = np.array(["R"] * 4)
    out = min_ev_bets(
        rid,
        p_win=np.array([0.65, 0.30, 0.62, 0.05]),
        odds_win=np.array([1.8, 2.0, 1.5, 30.0]),
        p_place=np.array([0.90, 0.70, 0.95, 0.30]),
        place_odds=np.array([1.1, 1.6, 1.2, 5.0]),
        min_ev=1.0, min_prob=0.6)
    # 0: 単 1.17 vs 複 0.99 -> 低い方は複勝(0.99<1.0で不成立) / 1: 単0.6 vs 複1.12 -> 低い方は単勝(0.6<1.0で不成立)
    # 2: 単0.93 vs 複1.14 -> 低い方は単勝(0.93<1.0で不成立) / 3: 単0.15 vs 複1.5 -> 低い方は単勝(不成立)
    assert list(out["bet_type"]) == [None, None, None, None]
    assert (out["stake_yen"] == 0).all()


def test_min_ev_bets_can_still_bet_when_the_lower_side_clears_the_bar():
    rid = np.array(["R", "R"])
    out = min_ev_bets(rid, p_win=np.array([0.70, 0.62]), odds_win=np.array([1.6, 1.7]),
                      p_place=np.array([0.90, 0.95]), place_odds=np.array([1.3, 1.25]),
                      min_ev=1.0, min_prob=0.6)
    # 単: 0.70*1.6=1.12 / 0.62*1.7=1.054 、複: 0.90*1.3=1.17 / 0.95*1.25=1.1875
    # 低い方はどちらも単勝で、EV>=1・p_win>=0.6 なので両方ベット成立
    assert list(out["bet_type"]) == ["単勝", "単勝"]
    assert (out["stake_yen"] > 0).all()


def test_min_ev_bets_falls_back_to_win_when_place_odds_are_missing():
    out = min_ev_bets(np.array(["R", "R"]), np.array([0.7, 0.3]), np.array([1.6, 3.0]),
                      np.array([0.9, 0.6]), np.array([np.nan, np.nan]))
    assert out["bet_type"].iloc[0] == "単勝"
    assert np.isnan(out["ev_place"]).all()


def test_fallback_bets_the_high_side_when_it_already_clears_both_bars():
    """高い方が確率もEVも満たすなら、フォールバックせず高い方のまま（max_ev_bets と同じ）。"""
    rid = np.array(["R"])
    out = max_ev_bets_with_fallback(rid, p_win=np.array([0.65]), odds_win=np.array([1.8]),
                                    p_place=np.array([0.90]), place_odds=np.array([1.1]),
                                    min_ev=1.0, min_prob=0.6)
    ref = max_ev_bets(rid, p_win=np.array([0.65]), odds_win=np.array([1.8]),
                      p_place=np.array([0.90]), place_odds=np.array([1.1]),
                      min_ev=1.0, min_prob=0.6)
    assert out["bet_type"].iloc[0] == ref["bet_type"].iloc[0] == "単勝"
    assert not out["fallback"].iloc[0]


def test_fallback_switches_to_the_low_side_when_only_probability_blocks_the_high_side():
    """高い方 EV=1.17（単勝, p=0.30<0.6で不成立）→ 低い方 EV=1.08（複勝, p=0.75≥0.6）に切り替え。"""
    rid = np.array(["R"])
    out = max_ev_bets_with_fallback(rid, p_win=np.array([0.30]), odds_win=np.array([3.9]),
                                    p_place=np.array([0.75]), place_odds=np.array([1.44]),
                                    min_ev=1.0, min_prob=0.6)
    assert out["ev_win"].iloc[0] > out["ev_place"].iloc[0]      # 単勝が「高い方」
    assert out["bet_type"].iloc[0] == "複勝"                     # だが賭けたのは低い方
    assert out["fallback"].iloc[0]
    assert out["stake_yen"].iloc[0] > 0
    assert out["ev_chosen"].iloc[0] == pytest.approx(out["ev_place"].iloc[0])


def test_fallback_does_not_bet_when_the_high_side_fails_on_ev_not_probability():
    """高い方が EV 不足で見送りのときは、低い方は必ずさらに EV が低いので賭けない。"""
    rid = np.array(["R"])
    out = max_ev_bets_with_fallback(rid, p_win=np.array([0.30]), odds_win=np.array([2.5]),
                                    p_place=np.array([0.75]), place_odds=np.array([1.2]),
                                    min_ev=1.0, min_prob=0.6)
    assert out["ev_win"].iloc[0] < 1.0    # 単勝(高い方) EV = 0.75 < 1.0
    assert out["bet_type"].iloc[0] is None
    assert not out["fallback"].iloc[0]


def test_fallback_does_not_bet_when_the_low_side_also_fails_probability():
    rid = np.array(["R"])
    out = max_ev_bets_with_fallback(rid, p_win=np.array([0.30]), odds_win=np.array([3.9]),
                                    p_place=np.array([0.50]), place_odds=np.array([2.16]),
                                    min_ev=1.0, min_prob=0.6)
    # 単勝(高い方) EV=1.17・p=0.30<0.6 不成立 → 複勝(低い方) EV=1.08・p=0.50<0.6 も不成立
    assert out["bet_type"].iloc[0] is None
    assert not out["fallback"].iloc[0]


def test_fallback_matches_max_ev_bets_when_place_data_is_missing():
    out = max_ev_bets_with_fallback(np.array(["R", "R"]), np.array([0.7, 0.3]),
                                    np.array([1.6, 3.0]), np.array([0.9, 0.6]),
                                    np.array([np.nan, np.nan]))
    ref = max_ev_bets(np.array(["R", "R"]), np.array([0.7, 0.3]), np.array([1.6, 3.0]),
                      np.array([0.9, 0.6]), np.array([np.nan, np.nan]))
    assert list(out["bet_type"]) == list(ref["bet_type"])
    assert not out["fallback"].any()
