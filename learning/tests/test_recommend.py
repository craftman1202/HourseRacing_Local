"""全券種の推奨買い目（nar.eval.recommend）。"""

from __future__ import annotations

import numpy as np
import pytest

from nar.eval.recommend import Rule, blended_probability, recommend, structures, tickets_for

TABLE6 = {"1_6": 0.55, "2_6": 0.18, "3_6": 0.11, "4_6": 0.08, "5_6": 0.05, "6_6": 0.03}
HORSE = np.arange(1, 7)
POP = np.array([2, 1, 3, 4, 5, 6], dtype=float)          # 2番が1番人気
WAKU = np.array([1, 2, 3, 4, 5, 5], dtype=float)


def test_lambda_one_is_the_popularity_table():
    p = blended_probability(np.full(6, 1 / 6), POP, TABLE6, 1.0)
    assert np.isclose(p.sum(), 1) and p.argmax() == 1
    assert np.isclose(p[1], 0.55)


def test_lambda_zero_is_the_model():
    pm = np.array([.1, .1, .5, .1, .1, .1])
    assert np.allclose(blended_probability(pm, POP, TABLE6, 0.0), pm)


def test_trifecta_rule_matches_the_production_fav_axis_rule():
    from nar.eval.exotic import trifecta_fav_axis
    [plan] = recommend(HORSE, WAKU, np.full(6, 1 / 6), POP, TABLE6,
                       [Rule("3連単 1着固定流し 相手8頭", 1.0, 0.5, stake=True)])
    old = trifecta_fav_axis(HORSE, POP, TABLE6)
    assert plan.axis == old.axis == 2
    assert sorted(plan.tickets) == sorted(old.tickets) and plan.stake_yen == 2000


def test_below_min_p_no_plan():
    assert recommend(HORSE, WAKU, np.full(6, 1 / 6), POP, TABLE6,
                     [Rule("単勝 1位", 1.0, 0.6)]) == []


def test_unordered_tickets_are_sorted_and_deduplicated():
    t = tickets_for("馬連 BOX3頭", np.array([5, 2, 9]))
    assert t == [(2, 5), (5, 9), (2, 9)]
    # 枠: 同じ枠の2頭はゾロ目、重複は1点
    t = tickets_for("枠連 軸流し 相手3頭", np.array([5, 5, 1, 1], dtype=object))
    assert t == [(5, 5), (1, 5)]


def test_bracket_needs_waku():
    r = [Rule("枠連 1点", 1.0, 0.0)]
    assert recommend(HORSE, None, np.full(6, 1 / 6), POP, TABLE6, r) == []
    [pl] = recommend(HORSE, WAKU, np.full(6, 1 / 6), POP, TABLE6, r)
    assert pl.tickets == [(1, 2)]                         # 人気1位=2番(枠2)、2位=1番(枠1)


def test_a_horse_without_popularity_is_treated_as_scratched():
    pop = POP.copy(); pop[3] = np.nan
    [pl] = recommend(HORSE, WAKU, np.full(6, 1 / 6), pop, TABLE6,
                     [Rule("3連単 1着固定流し 相手8頭", 1.0, 0.0)])
    assert 4 not in pl.partners and all(4 not in t for t in pl.tickets)
    assert len(pl.tickets) == 4 * 3                       # 残り5頭: 軸 + 相手4頭


def test_too_many_missing_popularity_means_no_plans():
    pop = POP.copy(); pop[:3] = np.nan
    assert recommend(HORSE, WAKU, np.full(6, 1 / 6), pop, TABLE6,
                     [Rule("単勝 1位", 1.0, 0.0)]) == []


@pytest.mark.parametrize("name", list(structures()))
def test_every_structure_builds_in_a_small_field(name):
    ids = np.arange(1, 6)
    t = tickets_for(name, ids)
    assert t and all(len(x) == len(structures()[name][1][0]) for x in t)
