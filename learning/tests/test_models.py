"""MD-01..11, EN-01..05: モデル層とアンサンブル。

TabM（MD-12..15）と階層ベイズ（MD-16..22）は未実装のため、対応するテストは置いていない。
存在しない実装に対する green は最も危険な緑なので、skip も置かない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nar import synth
from nar.eval import metrics
from nar.models import baselines
from nar.models.base import masked_softmax, to_batch
from nar.models.clogit import ConditionalLogit
from nar.models.ensemble import ConstrainedStacker, OOFViolation
from nar.models.lgbm import LgbmRanker, graded_labels, group_sizes

FEATS = list(synth.FEATURE_COLS)


@pytest.fixture(scope="module")
def batch(synth_tables):
    df = synth_tables["entry"].sort_values(["race_id", "horse_no"]).reset_index(drop=True)
    return to_batch(df, FEATS), df


@pytest.fixture(scope="module")
def fitted(batch):
    b, _ = batch
    return ConditionalLogit(l2=1e-6).fit(b)


# --------------------------------------------------------------------- MD-01/02
def test_md01_probabilities_sum_to_one_within_each_race(fitted, batch):
    b, _ = batch
    p = fitted.predict_proba(b)
    sums = (p * b.mask).sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-9)
    assert np.allclose(p[~b.mask], 0.0), "パディング位置に確率が漏れています"


def test_md02_zero_features_give_uniform_distribution(batch):
    b, _ = batch
    model = ConditionalLogit()
    model.beta = np.zeros(b.x.shape[2])
    model.feature_names = b.feature_names
    p = model.predict_proba(b)
    n = b.mask.sum(axis=1)
    expected = np.where(b.mask, 1.0 / n[:, None], 0.0)
    assert np.allclose(p, expected, atol=1e-9)


# ------------------------------------------------------------------------ MD-03
def test_md03_analytic_gradient_matches_numerical(batch):
    b, _ = batch
    small = to_batch(pd.DataFrame({
        **{c: np.random.default_rng(0).normal(size=200) for c in FEATS},
        "race_id": np.repeat(np.arange(20), 10),
        "is_win": np.tile([1] + [0] * 9, 20),
    }), FEATS)

    model = ConditionalLogit(l2=1e-3, l1=1e-3)
    beta = np.random.default_rng(1).normal(scale=0.3, size=len(FEATS))
    _, grad = model._nll_and_grad(beta, small)

    eps = 1e-6
    num = np.empty_like(grad)
    for i in range(len(beta)):
        up, dn = beta.copy(), beta.copy()
        up[i] += eps
        dn[i] -= eps
        num[i] = (model._nll_and_grad(up, small)[0] - model._nll_and_grad(dn, small)[0]) / (2 * eps)

    rel = np.abs(grad - num) / np.maximum(np.abs(num), 1e-8)
    assert rel.max() < 1e-5, f"最大相対誤差 {rel.max():.2e}"


# ------------------------------------------------------------------------ MD-04
@pytest.mark.slow
def test_md04_recovers_true_beta_on_synthetic_data(synth_large):
    """FX-03 の真の β を回収できること。

    判定は Wald の**同時**検定で行う。当初は「5係数すべてが ±2SE 以内」と
    していたが、これは 0.95^5 ≈ 77% しか通らない基準で、推定が正しくても
    約4回に1回落ちる（実際に落ちた）。係数ごとの被覆を数えると、係数の数が
    増えるほど厳しくなるという意味の無い性質も付いてくる。

    W = (β̂ − β)' I (β̂ − β) は帰無仮説の下で χ²_k に従う。ここでは I の
    対角のみ（= Σ z²）を使う近似で、有意水準 0.1% で判定する。
    併せて、1つの係数だけが大きく外れる壊れ方も見るため |z| の上限も置く。
    """
    from scipy import stats

    df = synth_large["entry"].sort_values(["race_id", "horse_no"]).reset_index(drop=True)
    b = to_batch(df, FEATS)
    model = ConditionalLogit(l2=1e-8).fit(b)
    se = model.standard_errors(b)
    true = np.asarray(synth.SynthConfig().beta)

    z = (model.beta - true) / se
    wald = float(np.sum(z ** 2))
    critical = float(stats.chi2.ppf(0.999, len(FEATS)))
    detail = {f: (round(float(e), 3), round(float(s), 3), float(t), round(float(zz), 2))
              for f, e, s, t, zz in zip(FEATS, model.beta, se, true, z)}
    assert wald < critical, f"Wald {wald:.1f} >= χ²_{len(FEATS)}(0.999)={critical:.1f}: {detail}"
    assert np.abs(z).max() < 4.0, f"1係数だけ大きく外れています: {detail}"


# ------------------------------------------------------------------------ MD-05
def test_md05_padding_values_do_not_affect_output(fitted, batch):
    b, _ = batch
    p_before = fitted.predict_proba(b)

    tampered = type(b)(b.x.copy(), b.y, b.mask, b.race_ids, b.row_index, b.feature_names)
    tampered.x[~b.mask] = 1e6
    p_after = fitted.predict_proba(tampered)

    assert np.allclose(p_before[b.mask], p_after[b.mask], atol=1e-9)


# ------------------------------------------------------------------------ MD-06
def test_md06_strong_l1_shrinks_all_coefficients_to_zero(batch):
    b, _ = batch
    model = ConditionalLogit(l2=0.0, l1=1e4).fit(b)
    assert np.abs(model.beta).max() < 1e-3, model.beta


def test_intercept_is_rejected():
    with pytest.raises(ValueError, match="切片"):
        ConditionalLogit(fit_intercept=True)


# --------------------------------------------------------------------- MD-07..10
def test_md07_md08_group_array_matches_rows_and_races(synth_tables):
    df = synth_tables["entry"].sort_values(["race_id", "horse_no"]).reset_index(drop=True)
    g = group_sizes(df)
    assert g.sum() == len(df)
    assert len(g) == df["race_id"].nunique()


def test_md08_interleaved_races_are_rejected(synth_tables):
    """レースが行方向で分断されていたら group を作らせない。"""
    df = synth_tables["entry"].sort_values("horse_no").reset_index(drop=True)
    with pytest.raises(ValueError, match="分断"):
        group_sizes(df)


def test_md09_graded_labels_are_in_range_with_one_top_grade(synth_tables):
    df = synth_tables["entry"]
    lab = graded_labels(df["finish_pos"])
    assert set(np.unique(lab)) <= {0, 1, 2, 3}
    per_race = pd.Series(lab).groupby(df["race_id"].to_numpy()).apply(lambda s: (s == 3).sum())
    assert (per_race == 1).all(), "各レースにラベル3がちょうど1つ（同着は別途例外処理）"


@pytest.mark.slow
def test_md10_lambdarank_softmax_sums_to_one(synth_tables):
    df = synth_tables["entry"].sort_values(["race_id", "horse_no"]).reset_index(drop=True)
    model = LgbmRanker(num_boost_round=30).fit(df, FEATS)
    p = model.predict_proba(df)
    sums = pd.Series(p).groupby(df["race_id"].to_numpy()).sum()
    assert np.allclose(sums, 1.0, atol=1e-9)


# ------------------------------------------------------------------- EN-01..05
@pytest.fixture
def oof_predictions(synth_tables):
    df = synth_tables["entry"].sort_values(["race_id", "horse_no"]).reset_index(drop=True)
    rid = df["race_id"].to_numpy()
    rng = np.random.default_rng(0)
    good = metrics.normalize_within_race(
        np.exp(df[FEATS].to_numpy() @ np.asarray(synth.SynthConfig().beta)), rid)
    noisy = metrics.normalize_within_race(good + rng.uniform(0, 0.05, len(df)), rid)
    uni = baselines.uniform(df)
    preds = pd.DataFrame({"good": good, "noisy": noisy, "uniform": uni})
    return preds, df["is_win"].to_numpy(), rid, df


def test_en01_weights_are_nonnegative_and_sum_to_one(oof_predictions):
    preds, y, rid, _ = oof_predictions
    st = ConstrainedStacker(list(preds.columns)).fit(preds, y, rid)
    assert (st.weights >= 0).all()
    assert st.weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_en02_in_fold_predictions_are_blocked(oof_predictions):
    preds, y, rid, _ = oof_predictions
    fold_ids = np.array([np.nan] * len(preds))
    with pytest.raises(OOFViolation, match="in-fold"):
        ConstrainedStacker(list(preds.columns)).fit(preds, y, rid, fold_ids=fold_ids)


def test_en03_all_weight_on_one_model_reproduces_that_model(oof_predictions):
    preds, y, rid, _ = oof_predictions
    st = ConstrainedStacker(list(preds.columns))
    st.weights = np.array([1.0, 0.0, 0.0])
    out = st.predict_proba(preds, rid)
    assert np.allclose(out, preds["good"].to_numpy(), atol=1e-9)


def test_en04_ensemble_is_not_worse_than_best_single_model(oof_predictions):
    preds, y, rid, _ = oof_predictions
    st = ConstrainedStacker(list(preds.columns)).fit(preds, y, rid)
    ens = metrics.race_nll(st.predict_proba(preds, rid), y, rid)
    best = min(metrics.race_nll(preds[c].to_numpy(), y, rid) for c in preds.columns)
    assert ens <= best + 0.005, f"アンサンブル {ens:.4f} > 最良単体 {best:.4f}"


def test_en05_ensemble_output_sums_to_one(oof_predictions):
    preds, y, rid, _ = oof_predictions
    st = ConstrainedStacker(list(preds.columns)).fit(preds, y, rid)
    out = pd.Series(st.predict_proba(preds, rid)).groupby(rid).sum()
    assert np.allclose(out, 1.0, atol=1e-9)


def test_masked_softmax_ignores_padding():
    scores = np.array([[1.0, 2.0, 99.0]])
    mask = np.array([[True, True, False]])
    p = masked_softmax(scores, mask)
    assert p[0, 2] == 0.0
    assert p[0].sum() == pytest.approx(1.0)
