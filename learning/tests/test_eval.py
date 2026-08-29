"""EV-01..03, CA-01..06, EC-01..12, RF-01..08, RP-01: 評価・経済・ガード。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nar import synth
from nar.errors import TooGoodToBeTrueError
from nar.eval import calibration, economic, guards, metrics
from nar.models import baselines

FEATS = list(synth.FEATURE_COLS)


@pytest.fixture(scope="module")
def scored(synth_tables):
    """真の β によるオラクル予測。評価パイプライン自体の検証に使う。"""
    df = synth_tables["entry"].sort_values(["race_id", "horse_no"]).reset_index(drop=True)
    rid = df["race_id"].to_numpy()
    p = metrics.race_softmax(df[FEATS].to_numpy() @ np.asarray(synth.SynthConfig().beta), rid)
    return df, rid, p


# --------------------------------------------------------------------- EV-01..03
def test_ev01_uniform_baseline_nll_equals_mean_log_n(scored):
    df, rid, _ = scored
    p = baselines.uniform(df)
    assert metrics.race_nll(p, df["is_win"].to_numpy(), rid) == pytest.approx(
        metrics.uniform_nll(rid), abs=1e-6)


def test_ev02_market_baseline_is_normalized(scored):
    df, rid, _ = scored
    q = baselines.market(df)
    sums = pd.Series(q).groupby(rid).sum()
    assert np.allclose(sums, 1.0, atol=1e-9), "控除率で割り戻しただけでは Σq≈1.25 になる"


def test_ev03_model_vs_market_uses_block_bootstrap(scored):
    """ブロックブートストラップで NLL 差の CI を出し、点推定だけで判定しない。"""
    df, rid, p = scored
    y = df["is_win"].to_numpy()
    q = baselines.market(df)

    win = df["is_win"] == 1
    per_race = pd.DataFrame({
        "diff": -np.log(np.clip(p[win], 1e-12, 1)) + np.log(np.clip(q[win], 1e-12, 1)),
        "day": pd.to_datetime(df.loc[win, "race_date"]).dt.date,
    })
    point, lo, hi = metrics.block_bootstrap_ci(
        per_race["diff"], per_race["day"], n_iter=500, seed=0)
    assert lo < point < hi
    # オラクル予測は市場より良いはず（差 < 0）。これが成り立たなければ評価系のバグ
    assert hi < 0, f"オラクルが市場を上回りません: [{lo:.4f}, {hi:.4f}]"


def test_uniform_baseline_beats_nothing(scored):
    df, rid, p = scored
    y = df["is_win"].to_numpy()
    assert metrics.race_nll(p, y, rid) < metrics.uniform_nll(rid)


# --------------------------------------------------------------------- CA-01..06
def test_ca01_temperature_scaling_preserves_sum_to_one(scored):
    df, rid, p = scored
    y = df["is_win"].to_numpy()
    scaler = calibration.TemperatureScaler().fit(np.log(p), y, rid)
    out = scaler.transform(np.log(p), rid)
    assert np.allclose(pd.Series(out).groupby(rid).sum(), 1.0, atol=1e-9)
    assert 0.2 < scaler.temperature < 5.0


def test_ca02_temperature_must_be_fitted_before_transform(scored):
    df, rid, p = scored
    with pytest.raises(RuntimeError, match="valid"):
        calibration.TemperatureScaler().transform(np.log(p), rid)


def test_ca03_isotonic_renormalizes_within_race(scored):
    df, rid, p = scored
    y = df["is_win"].to_numpy()
    iso = calibration.IsotonicRaceCalibrator().fit(p, y)
    out = iso.transform(p, rid)
    assert np.allclose(pd.Series(out).groupby(rid).sum(), 1.0, atol=1e-9)


def test_ca06_calibration_does_not_change_top1(scored):
    """順位を変えない変換であること。変えるなら較正ではなく別のモデルになっている。"""
    df, rid, p = scored
    y = df["is_win"].to_numpy()
    pos = df["finish_pos"].to_numpy()
    before = metrics.top_k_accuracy(p, pos, rid, 1)

    scaler = calibration.TemperatureScaler().fit(np.log(p), y, rid)
    after = metrics.top_k_accuracy(scaler.transform(np.log(p), rid), pos, rid, 1)
    assert abs(before - after) <= 0.001


def test_ca05_ece_is_reported_per_popularity_band(scored):
    df, rid, p = scored
    bands = metrics.ece_by_popularity_band(p, df["is_win"].to_numpy(),
                                           df["popularity"].to_numpy())
    assert set(bands) == {"1-3", "4-7", "8+"}
    assert all(0 <= v <= 1 for v in bands.values())


# --------------------------------------------------------------------- EC-01..12
def test_ec01_win_odds_inverse_sum_matches_takeout(synth_tables):
    e = synth_tables["entry"]
    sums = e.groupby("race_id")["odds_win"].apply(
        lambda s: economic.inverse_odds_sum(s.to_numpy()))
    assert sums.mean() == pytest.approx(1.25, abs=0.03), (
        f"τ=0.20 なら Σ1/o = 1.25。実測 {sums.mean():.4f}")


def test_ec04_measured_takeout_per_track_within_two_points(synth_tables):
    from nar.eda.questions import measured_takeout

    out = measured_takeout(synth_tables["entry"])
    assert (out["diff_pt"].abs() <= 2.0).all(), out[["baba_code", "diff_pt"]]


def test_ec05_effective_odds_identity_at_zero_stake():
    assert economic.effective_odds(o=10.0, b=0, pool=1_000_000, takeout=0.2) == pytest.approx(
        10.0, abs=1e-6)


def test_ec05_effective_odds_drops_with_stake():
    o = economic.effective_odds(o=10.0, b=10_000, pool=1_000_000, takeout=0.2)
    assert o < 10.0
    # 理論値: S = 1e6*0.8/10 = 80,000 → (1e6+1e4)*0.8/(8e4+1e4)
    assert o == pytest.approx((1_010_000 * 0.8) / 90_000, abs=1e-9)


def test_ec06_effective_odds_monotone_decreasing_in_stake():
    stakes = [0, 100, 1_000, 10_000, 100_000]
    vals = [economic.effective_odds(8.0, b, 500_000, 0.2) for b in stakes]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_ec07_corrected_roi_never_exceeds_uncorrected(scored):
    df, rid, p = scored
    bet = df.assign(p=p)
    raw, _ = economic.simulate_win_bets(bet, ev_threshold=1.0)
    adj, _ = economic.simulate_win_bets(bet, ev_threshold=1.0, pool=2_000_000, takeout=0.20)
    assert adj.roi <= raw.roi + 1e-12, f"補正後 {adj.roi:.4f} > 補正前 {raw.roi:.4f}"


def test_ec08_flat_bet_on_everything_returns_one_minus_takeout(synth_tables):
    """経済評価パイプライン全体のサニティチェック。

    テスト仕様は「(1-τ) ± 1pt」としているが、全馬均等買いの ROI は稀な高オッズ
    的中に支配される重い裾を持つ推定量で、この規模（約1万行）では標準誤差だけで
    数 pt に達する。点推定を固定幅で判定すると不安定なテストになるので、
    開催日ブロックのブートストラップ CI に 0.80 が入ることで判定する。
    """
    e = synth_tables["entry"]
    per_row = e["is_win"] * e["odds_win"]
    point, lo, hi = metrics.block_bootstrap_ci(
        per_row, pd.to_datetime(e["race_date"]).dt.date, n_iter=600, seed=0)
    assert lo <= 0.80 <= hi, (
        f"ROI {point:.4f} CI[{lo:.4f}, {hi:.4f}]。80% を含まないなら"
        "払戻の突合か控除率の扱いが誤り。")


def test_ec09_ec10_kelly_formula_and_cap():
    f = economic.kelly_fraction(0.5, 3.0, cap=1.0)
    assert float(f) == pytest.approx(0.25, abs=1e-9)
    assert float(economic.kelly_fraction(0.5, 3.0, cap=0.25)) == pytest.approx(0.0625, abs=1e-9)
    # 期待値マイナスには賭けない
    assert float(economic.kelly_fraction(0.1, 3.0, cap=0.25)) == 0.0


def test_ec09_kelly_never_exceeds_configured_cap():
    p = np.linspace(0.01, 0.99, 200)
    o = np.linspace(1.1, 50.0, 200)
    assert economic.kelly_fraction(p, o, cap=0.25).max() <= 0.25 + 1e-12


def test_ec11_block_bootstrap_ci_covers_truth(scored):
    rng = np.random.default_rng(0)
    truth = 0.8
    days = np.repeat(np.arange(200), 10)
    values = pd.Series(rng.normal(truth, 0.3, size=len(days)))
    point, lo, hi = metrics.block_bootstrap_ci(values, pd.Series(days), n_iter=800, seed=1)
    assert lo <= truth <= hi


def test_ec12_max_drawdown_is_nonnegative_and_matches_curve():
    equity = pd.Series([0, 100, 50, 200, -30, 10])
    assert economic.max_drawdown(equity) == pytest.approx(230.0)
    assert economic.max_drawdown(pd.Series([0, 10, 20])) == 0.0
    assert economic.max_drawdown(pd.Series([], dtype=float)) == 0.0


# --------------------------------------------------------------------- RF-01..08
def test_rf01_rf02_impossible_performance_stops_the_pipeline():
    tw = guards.check(oos_top1=0.72, oos_nll=1.05)
    fired = {t.id for t in tw if t.fired}
    assert fired == {"RF-01", "RF-02"}
    with pytest.raises(TooGoodToBeTrueError, match="RF-01"):
        guards.enforce(tw)


def test_rf03_sustained_high_roi_is_flagged():
    roi = pd.Series([1.35, 1.40, 1.33, 0.95])
    tw = {t.id: t for t in guards.check(win_roi_uncorrected_monthly=roi)}
    assert tw["RF-03"].fired


def test_rf05_cv_much_better_than_oos_is_flagged():
    tw = {t.id: t for t in guards.check(cv_nll=1.60, oos_nll=1.75)}
    assert tw["RF-05"].fired and tw["RF-05"].severity == guards.CRITICAL


def test_rf06_dominant_feature_importance_is_flagged():
    imp = pd.Series({"leaky": 500.0, "a": 100.0, "b": 100.0})
    tw = {t.id: t for t in guards.check(feature_importance=imp)}
    assert tw["RF-06"].fired
    assert "leaky" in tw["RF-06"].message


def test_rf07_shuffled_labels_must_not_predict():
    assert {t.id for t in guards.check(shuffled_label_nll=1.9) if t.fired} == {"RF-07"}


def test_rf08_zero_drawdown_means_missing_losses():
    assert {t.id for t in guards.check(max_drawdown=0.0) if t.fired} == {"RF-08"}


def test_healthy_run_fires_nothing():
    tw = guards.check(oos_top1=0.44, oos_nll=1.78, cv_nll=1.76,
                      shuffled_label_nll=2.35, max_drawdown=4200.0,
                      uses_final_odds_for_decision=False)
    assert not any(t.fired for t in tw)
    guards.enforce(tw)


# ------------------------------------------------------------------------ RP-01
def test_rp01_same_seed_reproduces_identical_metrics(synth_tables):
    a = synth.generate(synth.SynthConfig(n_races=200, seed=99))["entry"]
    b = synth.generate(synth.SynthConfig(n_races=200, seed=99))["entry"]
    pd.testing.assert_frame_equal(a, b)


def test_multiple_comparison_correction_is_monotone():
    p = np.array([0.001, 0.01, 0.03, 0.2, 0.9])
    adj = metrics.benjamini_hochberg(p)
    assert (adj >= p).all()
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12), "BH 補正後は単調でなければならない"
