"""特徴量選択・HPO・再現性記録。

CV-09/10, RP-02/03 に対応する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nar import synth
from nar.config import cv_config, feature_config
from nar.eval.splits import inner_folds, make_folds
from nar.features import selection

FEATS = list(synth.FEATURE_COLS)


@pytest.fixture(scope="module")
def train_df(synth_tables):
    d = synth_tables["entry"].sort_values(["race_id", "horse_no"]).reset_index(drop=True)
    return d


# ------------------------------------------------------------- 第1段 Null Importance
def test_null_importance_keeps_real_signal(train_df):
    ni = selection.null_importance(train_df, FEATS, n_runs=8, seed=0)
    top = ni.iloc[0]
    assert top["feature"] == "x_speed", "真の係数が最大の特徴量が首位に来ていません"
    assert bool(top["survives"])


def test_null_importance_drops_pure_noise(train_df):
    d = train_df.copy()
    rng = np.random.default_rng(0)
    d["x_noise"] = rng.normal(size=len(d))
    ni = selection.null_importance(d, FEATS + ["x_noise"], n_runs=12, seed=0)
    noise = ni[ni["feature"] == "x_noise"].iloc[0]
    assert not bool(noise["survives"]), (
        f"純粋なノイズが生き残りました（実 {noise['actual_gain']:.1f} / 帰無95% "
        f"{noise['null_p95']:.1f}）")


def test_null_importance_flags_a_leak_column(train_df):
    """FX-04 のリーク列は帰無分布を大きく超えるはず（LK-08）。"""
    d = synth.inject_leak_column(train_df)
    ni = selection.null_importance(d, FEATS + ["x_leak"], n_runs=8, seed=0)
    assert ni.iloc[0]["feature"] == "x_leak"
    assert ni.iloc[0]["actual_gain"] > ni.iloc[0]["null_p95"] * 2


def test_label_shuffle_stays_within_race(train_df):
    rng = np.random.default_rng(0)
    shuffled = selection.shuffle_labels_within_race(train_df, rng)
    per_race = pd.DataFrame({"r": train_df["race_id"], "p": shuffled})
    # レース内の着順の集合は保存される（頭数分布を壊さない）
    orig = train_df.groupby("race_id")["finish_pos"].apply(lambda s: sorted(s.tolist()))
    new = per_race.groupby("r")["p"].apply(lambda s: sorted(s.tolist()))
    assert (orig == new).all()


# -------------------------------------------------------------- 第2段 相関・VIF
def test_drop_collinear_removes_a_duplicated_column(train_df):
    d = train_df.copy()
    d["x_speed_copy"] = d["x_speed"] * 2.0 + 0.1
    keep, dropped = selection.drop_collinear(d, FEATS + ["x_speed_copy"])
    assert "x_speed_copy" in dropped or "x_speed" in dropped
    assert len({"x_speed", "x_speed_copy"} & set(keep)) == 1


def test_vif_is_finite_for_independent_features(train_df):
    vif = selection.variance_inflation(train_df, FEATS)
    assert np.isfinite(vif).all()
    assert (vif < 5).all(), f"独立なはずの特徴量で VIF が高すぎます:\n{vif}"


# -------------------------------------------------------------------- 全体
def test_select_runs_all_three_stages(train_df):
    res = selection.select(train_df, FEATS, n_null_runs=6, apply_vif=True, run_rfe=False)
    assert set(res.selected) <= set(FEATS)
    assert len(res.selected) >= 1
    assert len(res.null_importance) == len(FEATS)


def test_cv10_selection_statistics_differ_between_folds(synth_tables):
    """fold ごとに統計量が異なること。全 fold 同一なら全期間実行のバグ。"""
    d = synth_tables["entry"]
    a = selection.null_importance(d[d["race_date"] < "2016-01-01"], FEATS, n_runs=5, seed=0)
    b = selection.null_importance(d[d["race_date"] >= "2016-01-01"], FEATS, n_runs=5, seed=0)
    merged = a.merge(b, on="feature", suffixes=("_a", "_b"))
    assert not np.allclose(merged["actual_gain_a"], merged["actual_gain_b"]), (
        "fold 間で importance が完全一致しています")


# ---------------------------------------------------------------------- HPO
def test_cv09_inner_folds_stay_inside_outer_train():
    ccfg = cv_config()
    outer = make_folds(ccfg)[-1]
    for f in inner_folds(outer, ccfg.inner_n_folds, ccfg.embargo_days):
        assert outer.train_start <= f.train_start
        assert f.valid_end <= outer.train_end


def test_hpo_search_improves_over_random_and_records_trials(synth_tables):
    from nar.eval.metrics import race_softmax
    from nar.train import hpo

    ccfg = cv_config()
    feat = synth_tables["entry"].copy()
    outer = make_folds(ccfg)[-1]

    def train_predict(params, tr, va, cols):
        # l2 が大きいほど係数が 0 に潰れて一様分布に近づく、という単純な依存を作る
        w = np.array([1.0, 0.5, 0.3, 0.0, 0.0]) / (1.0 + params["l2"] * 100)
        return race_softmax(va[cols].to_numpy() @ w, va["race_id"].to_numpy())

    res = hpo.run("clogit", outer, feat, FEATS, train_predict, n_trials=12,
                  embargo_days=ccfg.embargo_days, n_inner=ccfg.inner_n_folds)
    assert res.n_trials == 12
    assert res.best_value < 2.3, "一様分布より良い解が見つかっていません"
    assert len(res.trials) == 12
    assert "l2" in res.best_params


def test_hpo_objective_is_aligned_with_prepared_valid_rows(synth_tables):
    """目的関数のラベルと予測が行単位で対応していること。

    prepare() は (race_id, horse_no) でソートする。以前は train_predict 側が
    内部で prepare し、目的関数はソート前の valid からラベルを取っていたため、
    HPO が実質シャッフルされたラベルを最適化していた。完全予測を返すダミーで
    NLL がほぼ 0 になることを確認し、その経路が復活したら落ちるようにする。
    """
    from nar.train import hpo
    from nar.train.pipeline import prepare

    ccfg = cv_config()
    outer = make_folds(ccfg)[-1]
    feat = synth_tables["entry"].copy()

    def oracle(params, tr, va, cols):
        # 勝ち馬に確率 1 を与える完全予測。行順が合っていれば NLL ≈ 0
        p = va["is_win"].to_numpy(dtype=float)
        return np.clip(p, 1e-9, None)

    res = hpo.run("clogit", outer, feat, FEATS, oracle, n_trials=2,
                  embargo_days=ccfg.embargo_days, n_inner=ccfg.inner_n_folds,
                  prepare_fn=prepare)
    assert res.best_value < 1e-6, (
        f"完全予測なのに NLL {res.best_value:.4f}。予測とラベルの行順がずれています。")


def test_hpo_rejects_prediction_length_mismatch(synth_tables):
    from nar.train import hpo
    from nar.train.pipeline import prepare

    ccfg = cv_config()
    outer = make_folds(ccfg)[-1]
    with pytest.raises(ValueError, match="行順"):
        hpo.run("clogit", outer, synth_tables["entry"], FEATS,
                lambda params, tr, va, cols: np.ones(len(va) - 1),
                n_trials=1, embargo_days=ccfg.embargo_days, prepare_fn=prepare)


def test_hpo_uses_a_different_search_sequence_per_fold(synth_tables):
    """fold ごとに探索列が変わること。全 fold 同一の best_params は fold 内実行の失敗徴候。"""
    from nar.eval.metrics import race_softmax
    from nar.train import hpo
    from nar.train.pipeline import prepare

    ccfg = cv_config()
    feat = synth_tables["entry"].copy()

    def train_predict(params, tr, va, cols):
        w = np.array([1.0, 0.5, 0.3, 0.0, 0.0]) / (1.0 + params["l2"] * 100)
        return race_softmax(va[cols].to_numpy() @ w, va["race_id"].to_numpy())

    seen = []
    for outer in make_folds(ccfg)[:2]:
        res = hpo.run("clogit", outer, feat, FEATS, train_predict, n_trials=6,
                      embargo_days=ccfg.embargo_days, n_inner=ccfg.inner_n_folds,
                      prepare_fn=prepare)
        seen.append(tuple(sorted(res.trials["params_l2"].round(10).tolist())))
    assert seen[0] != seen[1], "全 fold で同じ候補点を評価しています"


def test_hpo_rejects_windows_without_enough_data():
    from nar.train import hpo

    ccfg = cv_config()
    outer = make_folds(ccfg)[0]
    empty = pd.DataFrame({"race_id": [], "race_date": pd.to_datetime([]), "is_win": []})
    with pytest.raises(ValueError, match="十分なデータ"):
        hpo.run("clogit", outer, empty, FEATS, lambda *a: np.array([]),
                n_trials=2, embargo_days=ccfg.embargo_days)


# ---------------------------------------------------------------- RP-02/03
def test_rp02_run_record_is_complete(tmp_path):
    from nar.config import CONF_DIR
    from nar.tracking import RunTracker

    tracker = RunTracker("test-exp", tmp_path)
    with tracker.run("r1", "ds-abc123", CONF_DIR) as run:
        run.log_params({"model": "clogit", "l2": 1e-3})
        run.log_metrics({"race_nll": 2.05, "top1": 0.28})

    c = tracker.completeness()
    assert c["has_runs"] and c["git_commit"] and c["conf_snapshot"]
    assert c["dataset_version"] and c["metrics"]

    rec = tracker.runs[-1]
    assert set(rec["conf"]) >= {"cv.yaml", "features.yaml", "eda.yaml", "data.yaml"}
    assert rec["libraries"]["python"].startswith("3.")
    assert (tmp_path / "runs.json").exists()


def test_rp03_dataset_version_changes_when_raw_changes(tmp_path):
    from nar.io.manifest import Manifest, Record

    m = Manifest(tmp_path / "m.duckdb")
    m.upsert(Record(file_key="monthly/race/2026-07", sha256="aaa", status="ok"))
    v1 = m.dataset_version()
    m.upsert(Record(file_key="monthly/race/2026-07", sha256="bbb", status="ok"))
    v2 = m.dataset_version()
    assert v1 != v2, "raw が変わったのに dataset_version が変わっていません"


def test_tracking_survives_without_mlflow(tmp_path, monkeypatch):
    """記録基盤が無い環境でも実験を止めない（JSON へフォールバック）。"""
    import builtins

    from nar.config import CONF_DIR
    from nar.tracking import RunTracker

    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name == "mlflow":
            raise ImportError("no mlflow")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    tracker = RunTracker("exp", tmp_path)
    assert tracker.backend == "json"
    with tracker.run("r", "ds", CONF_DIR) as run:
        run.log_metrics({"nll": 1.0})
    assert tracker.completeness()["has_runs"]
