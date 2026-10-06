"""3連単「1番人気1着固定流し」の判定（nar.eval.exotic、2026-10-06）。"""

import numpy as np

from nar.eval.exotic import trifecta_fav_axis

T6 = {"1_6": 0.55, "2_6": 0.18, "3_6": 0.11, "4_6": 0.08, "5_6": 0.05, "6_6": 0.03}


def test_axis_is_the_favourite_and_partners_follow_popularity():
    p = trifecta_fav_axis(np.array([1, 2, 3, 4, 5, 6]), np.array([3, 1, 2, 4, 6, 5]), T6)
    assert p.axis == 2 and p.partners == [3, 1, 4, 6, 5]
    assert len(p.tickets) == 20 and len(set(p.tickets)) == 20 and p.stake_yen == 2000


def test_partners_are_capped_at_n_partners():
    t = {f"{r}_12": 0.6 if r == 1 else 0.4 / 11 for r in range(1, 13)}
    p = trifecta_fav_axis(np.arange(1, 13), np.arange(1, 13, dtype=float), t, n_partners=8)
    assert p.partners == list(range(2, 10)) and len(p.tickets) == 56


def test_returns_none_below_the_bar_or_with_missing_popularity():
    assert trifecta_fav_axis(np.arange(1, 7), np.arange(1, 7.0), {**T6, "1_6": 0.4}) is None
    assert trifecta_fav_axis(np.arange(1, 7), np.array([1, 2, np.nan, 4, 5, 6.0]), T6) is None
    assert trifecta_fav_axis(np.array([1, 2]), np.array([1.0, 2.0]), T6) is None
