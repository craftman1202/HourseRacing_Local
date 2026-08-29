"""TB-01..07: 二トラック構成。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nar import synth
from nar.eval import metrics
from nar.models import baselines
from nar.models.trackb import (
    MAX_ESTIMATED_PARAMS, ResidualOddsModel, align_to_odds_period, market_implied,
)


@pytest.fixture(scope="module")
def track_data(synth_tables):
    d = synth_tables["entry"].sort_values(["race_id", "horse_no"]).reset_index(drop=True)
    d["odds_change"] = 0.0
    rid = d["race_id"].to_numpy()
    # トラックA の出力に相当するもの（真の効用より弱い代理）
    p_a = metrics.race_softmax(d[["x_speed", "x_form"]].to_numpy() @ np.array([1.0, 0.5]), rid)
    q = market_implied(d)
    return d, p_a, q


# ------------------------------------------------------------------------ TB-01
def test_tb01_underpowered_warning_below_5000_races(track_data):
    d, p_a, q = track_data
    small = d[d["race_id"].isin(d["race_id"].drop_duplicates().head(300))]
    model = ResidualOddsModel(["odds_change"])
    rep = model.report(small)
    assert rep.n_races < 5000
    assert rep.underpowered is True
    assert any("検出力不足" in n for n in rep.notes)


def test_tb01_large_sample_clears_the_power_warning(track_data):
    d, _, _ = track_data
    big = pd.concat([d.assign(race_id=d["race_id"] + f"_{i}") for i in range(6)])
    rep = ResidualOddsModel().report(big)
    assert rep.n_races >= 5000 and rep.underpowered is False


# ------------------------------------------------------------------------ TB-02
def test_tb02_track_a_offset_coefficient_is_fixed_at_one(track_data):
    """log p^A の係数は学習対象パラメータに含まれない。"""
    d, p_a, q = track_data
    model = ResidualOddsModel(["odds_change"]).fit(d, p_a, q)
    # 推定されたのは η と α のみ
    assert model.n_estimated_params() == 2
    # オフセットを2倍にすると予測は変わる = 係数が 1.0 に固定されている証拠
    p1 = model.predict_proba(d, p_a, q)
    p2 = model.predict_proba(d, p_a ** 2, q)
    assert not np.allclose(p1, p2)


# ------------------------------------------------------------------------ TB-03
def test_tb03_estimated_parameter_count_is_capped(track_data):
    d, p_a, q = track_data
    assert ResidualOddsModel(["odds_change"]).n_estimated_params() <= MAX_ESTIMATED_PARAMS
    with pytest.raises(ValueError, match="以下に抑えて"):
        ResidualOddsModel([f"v{i}" for i in range(MAX_ESTIMATED_PARAMS)])


# ------------------------------------------------------------------------ TB-04
def test_tb04_eta_zero_reproduces_track_a_exactly(track_data):
    d, p_a, q = track_data
    model = ResidualOddsModel()
    model.eta = 0.0
    p = model.predict_proba(d, p_a, q)
    assert np.allclose(p, p_a, atol=1e-9), "η=0 でトラックA と一致しません"


def test_residual_model_improves_on_track_a_when_market_is_informative(track_data):
    d, p_a, q = track_data
    rid = d["race_id"].to_numpy()
    y = d["is_win"].to_numpy()
    model = ResidualOddsModel().fit(d, p_a, q)
    p_b = model.predict_proba(d, p_a, q)
    assert metrics.race_nll(p_b, y, rid) <= metrics.race_nll(p_a, y, rid) + 1e-9
    assert model.eta > 0, "市場情報が有効なら η は正になるはず"


# ------------------------------------------------------------------------ TB-05
def test_tb05_report_carries_reference_only_disclaimer(track_data):
    d, p_a, q = track_data
    rep = ResidualOddsModel().fit(d, p_a, q).report(d)
    joined = " ".join(rep.notes)
    assert "参考値" in joined
    assert "2年分" in joined


# ------------------------------------------------------------------------ TB-06
def test_tb06_comparison_is_restricted_to_the_odds_period(track_data):
    """トラックA を全期間、B を6か月で比較するのは無意味。突合対象を揃える。"""
    d, _, _ = track_data
    partial = d.copy()
    early = partial["race_date"] < partial["race_date"].quantile(0.5)
    partial.loc[early, "odds_win"] = np.nan

    (aligned,) = align_to_odds_period(partial)
    assert aligned["odds_win"].notna().all()
    assert aligned["race_id"].nunique() < partial["race_id"].nunique()


def test_tb06_alignment_intersects_across_frames(track_data):
    d, _, _ = track_data
    a = d.copy()
    b = d[d["race_id"].isin(d["race_id"].drop_duplicates().head(100))].copy()
    aa, bb = align_to_odds_period(a, b)
    assert set(aa["race_id"]) == set(bb["race_id"])


# ------------------------------------------------------------------------ TB-07
def test_tb07_daily_odds_snapshot_gap_is_detected():
    """開催日ごとにスナップショットが記録され、欠測3日以上で警告。"""
    from nar.models.trackb import snapshot_gaps

    days = pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-06", "2026-03-07"])
    gaps = snapshot_gaps(pd.Series(days), max_gap_days=3)
    assert len(gaps) == 1
    assert gaps.iloc[0]["gap_days"] == 4


def test_tb07_continuous_snapshots_produce_no_warning():
    from nar.models.trackb import snapshot_gaps

    days = pd.date_range("2026-03-01", periods=10, freq="D")
    assert len(snapshot_gaps(pd.Series(days), max_gap_days=3)) == 0


def test_market_implied_probabilities_sum_to_one(track_data):
    d, _, q = track_data
    assert np.allclose(pd.Series(q).groupby(d["race_id"].to_numpy()).sum(), 1.0, atol=1e-9)
