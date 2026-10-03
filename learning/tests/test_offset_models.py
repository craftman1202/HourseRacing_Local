"""残差モデル（offset_last）と回帰（task="regression"）— 2026-10-02 追加の任意機能。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nar.models.base import to_batch
from nar.models.clogit import ConditionalLogit
from nar.models.tabm import TabM, TabMConfig


def _frame(n_races=300, n=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        off = np.log(rng.dirichlet(np.ones(n)))
        x = rng.normal(size=n)
        logits = off + 0.8 * x
        p = np.exp(logits - logits.max()); p /= p.sum()
        w = rng.choice(n, p=p)
        for i in range(n):
            rows.append({"race_id": f"R{r:04d}", "horse_no": i + 1, "x": x[i], "offset": off[i],
                         "is_win": int(i == w), "y_reg": 2.0 * x[i] + rng.normal(scale=0.1)})
    return pd.DataFrame(rows)


def test_clogit_offset_is_fixed_at_coefficient_one_and_beta_recovers_the_residual():
    df = _frame()
    m = ConditionalLogit(l2=1e-6, offset_last=True).fit(to_batch(df, ["x", "offset"]))
    assert m.beta.shape == (1,)
    assert m.beta[0] == pytest.approx(0.8, abs=0.25)


def test_clogit_without_offset_is_unchanged():
    df = _frame()
    a = ConditionalLogit(l2=1e-3).fit(to_batch(df, ["x"]))
    b = ConditionalLogit(l2=1e-3, offset_last=False).fit(to_batch(df, ["x"]))
    assert np.allclose(a.beta, b.beta)


def test_tabm_offset_last_excludes_the_offset_from_the_network_and_adds_it_raw():
    df = _frame(n_races=60)
    cfg = TabMConfig(k=2, hidden=8, n_layers=1, epochs=1, batch_races=32, offset_last=True,
                     device="cpu")
    tm = TabM(cfg).fit(to_batch(df, ["x", "offset"]))
    assert tm.net.blocks[0].linear.in_features == 1
    assert tm._mu[-1] == 0.0 and tm._sd[-1] == 1.0
    b = to_batch(df, ["x", "offset"])
    p = tm.predict_proba(b)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_tabm_regression_learns_a_continuous_target():
    df = _frame(n_races=200)
    cfg = TabMConfig(k=2, hidden=16, n_layers=1, epochs=30, batch_races=32, lr=5e-3,
                     task="regression", device="cpu")
    b = to_batch(df, ["x"], label_col="y_reg")
    tm = TabM(cfg).fit(b)
    pred = b.flat_predictions(tm.predict_scores(b), len(df))
    assert np.corrcoef(pred, df["y_reg"])[0, 1] > 0.9


def test_tabm_rejects_unknown_task():
    with pytest.raises(ValueError):
        TabM(TabMConfig(task="poisson", device="cpu")).fit(to_batch(_frame(n_races=5), ["x"]))
