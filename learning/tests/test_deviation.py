"""人気との乖離（回帰）モデルの確率変換 nar.eval.deviation（2026-10-05）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nar.eval.deviation import (
    log_p_pop, pop_rank_inputs, popularity_from_odds, reg_probability,
)
from nar.eval.place import race_log_softmax
from nar.models.ensemble import ConstrainedStacker


def test_pop_rank_inputs_fill_missing_with_field_size_and_clip():
    assert list(pop_rank_inputs([1, np.nan, 20, 0], [8, 8, 18, 5])) == [1, 8, 16, 5]


def test_log_p_pop_normalizes_within_race_and_falls_back_to_uniform():
    rid = np.array(["A"] * 3 + ["B"] * 2)
    lp = log_p_pop([1, 2, 3, 1, 2], [3, 3, 3, 2, 2], rid, {"1_3": 0.6, "2_3": 0.3})
    p = np.exp(lp)
    assert p[:3].sum() == pytest.approx(1) and p[3:].sum() == pytest.approx(1)
    assert np.allclose(p[3:], 0.5), "表に無い組は 1/頭数"


def test_reg_probability_matches_the_stacked_blend_used_in_evaluation():
    rng = np.random.default_rng(0)
    rid = np.repeat(["a", "b"], [8, 10])
    off = race_log_softmax(rng.normal(size=18), rid)
    preds = {"ridge": rng.normal(size=18), "lgbm": rng.normal(size=18)}
    cal = {"ridge": {"gamma": 0.03, "temperature": 0.97},
           "lgbm": {"gamma": 0.11, "temperature": 0.9}}
    w = {"ridge": 0.2, "lgbm": 0.8}
    probs = {m: np.exp(race_log_softmax((off - cal[m]["gamma"] * preds[m])
                                        / cal[m]["temperature"], rid)) for m in preds}
    st = ConstrainedStacker(list(w))
    st.weights = np.array([w[m] for m in w])
    ref = st.predict_proba(pd.DataFrame(probs), rid)
    assert np.abs(reg_probability(off, preds, cal, w, rid) - ref).max() < 1e-12


def test_reg_probability_ignores_zero_weight_models_but_requires_weighted_ones():
    rid = np.array(["R"] * 3)
    cal = {"lgbm": {"gamma": 0.1, "temperature": 1.0}}
    p = reg_probability(np.log([0.5, 0.3, 0.2]), {"lgbm": np.zeros(3)}, cal,
                        {"lgbm": 1.0, "tabm": 0.0}, rid)
    assert np.allclose(p, [0.5, 0.3, 0.2])
    with pytest.raises(KeyError):
        reg_probability(np.zeros(3), {}, cal, {"lgbm": 1.0}, rid)


def test_popularity_from_odds_ties_share_the_minimum_rank():
    assert popularity_from_odds(pd.Series([2.0, 2.0, 5.0, -1])).tolist()[:3] == [1.0, 1.0, 3.0]
