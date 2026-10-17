"""全モデル（設計書 §8 の4モデル + CatBoost）の特徴量・HPO・walk-forward・アンサンブル（2026-10-13）。

方針（汎化性能を最優先）
  - ラベルは中止馬リークを塞いだ版: 競走中止・失格の馬は「走って負けた馬」として残す
    （pipeline.trainable は完走馬だけを残しており、レース後の情報が分母に漏れていた。
     memory: prediction-rows-exclude-dnf-horses-leak）。
  - HPO と特徴量の選択は **2019 年より前（fold1 の学習期間内）だけ**で1回行い、全 fold・OOS に
    同じ設定を使う。内側の検証は 2016/2017/2018 の各年（Feb〜Dec）、学習はその前 W 年（W も探索）。
    どの fold の検証期間とも重ならないので、選択バイアスが fold の成績に入らない。
    （設計書 §9.2 は fold ごとの nested HPO だが、500 試行 × 5 fold × 2 系統は数か月かかる。
     ここでは nested を「全 fold 共通の事前期間」に置き換え、より保守的にした）
  - 特徴量は 2010〜2018 で null importance 順位を作り、各モデルの HPO で「上位 n 個」の n を
    パラメータと同時に選ぶ（モデルごとに最適な特徴量数）。
  - walk-forward（fold1〜5 = 2019〜2023）: 学習期間の最後 180 日を較正・早期停止用に取り分け、
    残りで学習 → 温度スケーリングは取り分けた期間で（検証期間では合わせない）。
  - アンサンブルは OOF のみで重みを推定（ConstrainedStacker）。fold k の重みは fold < k から。
  - OOS（2024-02〜）は `oos` サブコマンドで、OOSGuard を開けたときだけ。

系統: pop = 人気＋カテゴリ（gold/features_noodds_pop_cat）、fund = そこから market_pop_* を除いたもの。

使い方:
  PYTHONPATH=src .venv/bin/python scripts/all_models.py rank pop
  PYTHONPATH=src .venv/bin/python scripts/all_models.py hpo pop lgbm --trials 500
  PYTHONPATH=src .venv/bin/python scripts/all_models.py fit pop lgbm
  PYTHONPATH=src .venv/bin/python scripts/all_models.py ensemble pop
  PYTHONPATH=src .venv/bin/python scripts/all_models.py oos pop --reason "..."
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nar.eval import metrics  # noqa: E402
from nar.eval.calibration import TemperatureScaler  # noqa: E402

log = logging.getLogger("all_models")
OUT = ROOT / "artifacts/all_models"
GOLD = ROOT / "data_real/gold/features_noodds_pop_cat/features.parquet"
CONF = ROOT / "conf_pop_cat"
MODELS = ("clogit", "lgbm", "catboost", "tabm", "bayes")
EMBARGO_DAYS = 31                     # valid_start の前 1 か月（cv.yaml の fold 境界と同じ形）
CAL_DAYS = 180                        # 学習期間末尾の較正・早期停止用
TUNE_VALID_YEARS = (2016, 2017, 2018)
RANK_YEARS = (2010, 2018)
FOLDS = {1: ("2019-02-01", "2019-12-31"), 2: ("2020-02-01", "2020-12-31"),
         3: ("2021-02-01", "2021-12-31"), 4: ("2022-02-01", "2022-12-31"),
         5: ("2023-02-01", "2023-12-31")}
OOS = ("2024-02-01", None)
POP_COLS = ("market_pop_rank", "market_pop_logrank", "market_pop_pct")
DNF_MARGINS = ("競走中止", "失格")
META = ["race_id", "horse_no", "race_date", "start_ts", "is_win", "finish_pos",
        "horse_sk", "jockey_sk", "sire_sk"]


# ====================================================================== データ
def feature_columns(variant: str) -> list[str]:
    os.environ["NAR_CONF_DIR"] = str(CONF)
    from nar.config import feature_config
    from nar.features.builder import asof_features

    cols = list(asof_features(feature_config(str(CONF))))
    if variant == "fund":
        cols = [c for c in cols if c not in POP_COLS]
    elif variant != "pop":
        raise ValueError(variant)
    return cols


_DATA: dict = {}


def load(variant: str) -> tuple[pd.DataFrame, list[str]]:
    """学習・評価に使う行（走った馬全員、勝者がちょうど1頭のレース）。"""
    if "df" not in _DATA:
        cache = OUT / "data_cache.parquet"
        if not cache.exists():
            build_cache(cache)
        # 2026-10-14: 4 プロセスが gold 全体（ピーク 15GB/プロセス）を同時に読んで WSL ごと
        # 落ちたため、整形済みの float32 キャッシュ（1 回だけ作る）を読む形にした
        df = pd.read_parquet(cache)
        df["race_date"] = pd.to_datetime(df["race_date"])
        _DATA["df"] = df
    cols = [c for c in feature_columns(variant) if c in _DATA["df"].columns]
    return _DATA["df"], cols


def build_cache(path: Path) -> None:
    cols_all = feature_columns("pop")
    import pyarrow.parquet as pq

    have = set(pq.read_schema(GOLD).names)
    df = pd.read_parquet(GOLD, columns=[c for c in META + cols_all if c in have])
    df["horse_no"] = df["horse_no"].astype(int)
    e = pd.read_parquet(ROOT / "data_real/silver/entry.parquet", columns=["race_id", "horse_no", "margin"])
    e["horse_no"] = e["horse_no"].astype(int)
    df = df.merge(e, on=["race_id", "horse_no"], how="left")
    dnf = df["finish_pos"].isna() & df["margin"].isin(DNF_MARGINS)
    ran = df["finish_pos"].notna() | dnf
    df = df[ran].copy()
    df["dnf"] = dnf[ran].astype(np.int8).to_numpy()
    df["is_win"] = df["is_win"].fillna(0)
    # 中止・失格は最下位扱い（LambdaRank 等の段階ラベル用）
    df["finish_pos"] = df["finish_pos"].fillna(99.0)
    wins = df.groupby("race_id")["is_win"].transform("sum")
    df = df[wins == 1].drop(columns=["margin"])
    for c in cols_all:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    df = df.sort_values(["race_id", "horse_no"]).reset_index(drop=True)
    log.info("行 %s / レース %s / 中止・失格 %s 頭", f"{len(df):,}", f"{df.race_id.nunique():,}",
             f"{int(df.dnf.sum()):,}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def fit_stats(train: pd.DataFrame, cols: list[str]) -> dict:
    from nar.train.pipeline import fit_stats as _fs
    return _fs(train, cols)


def standardize(df: pd.DataFrame, cols: list[str], stats: dict) -> pd.DataFrame:
    """凍結統計量で補完・標準化（行順は保つ。df は race_id, horse_no 順で来る）。"""
    out = df.copy()
    for c in cols:
        st = stats[c]
        out[c] = ((pd.to_numeric(out[c], errors="coerce").fillna(st["median"]) - st["mean"])
                  / (st["std"] or 1.0)).astype(np.float32)
    return out


def window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    m = (df["race_date"] >= pd.Timestamp(start)) & (df["race_date"] <= pd.Timestamp(end))
    return df[m]


def split_train(df: pd.DataFrame, valid_start: str, years: float | None):
    """valid_start − embargo までを学習、その末尾 CAL_DAYS を較正・早期停止に取り分ける。"""
    end = pd.Timestamp(valid_start) - pd.Timedelta(days=EMBARGO_DAYS)
    start = pd.Timestamp("1998-01-01") if years is None else end - pd.DateOffset(years=int(years))
    tr = window(df, start, end)
    cut = end - pd.Timedelta(days=CAL_DAYS)
    return tr[tr["race_date"] <= cut], tr[tr["race_date"] > cut]


# ====================================================================== モデル
KEEP_MODEL = False      # True のとき info["model"] に学習済みモデルを残す（配布物の作成用）


def fit_predict(model: str, params: dict, fit_df, cal_df, targets: list, cols: list[str]):
    """fit_df で学習（cal_df は早期停止）、targets（DataFrame のリスト）の確率を返す。

    戻り値: ([各 target の確率], info dict)。確率はレース内正規化済み・温度前。
    """
    p = dict(params)
    info: dict = {}
    if model == "clogit":
        from nar.models.base import to_batch
        from nar.models.clogit import ConditionalLogit

        m = ConditionalLogit(l2=p.get("l2", 1e-3), l1=p.get("l1", 0.0))
        m.fit(to_batch(fit_df, cols), max_iter=int(p.get("max_iter", 500)))
        outs = []
        for t in targets:
            b = to_batch(t, cols)
            outs.append(b.flat_predictions(m.predict_proba(b), len(t)))
        return outs, info

    if model == "lgbm":
        import lightgbm as lgb
        from nar.models.lgbm import graded_labels, group_sizes, label_gain_for

        grades = int(p.pop("_label_grades", 3))
        rounds = int(p.pop("_max_rounds", 3000))
        obj = p.pop("objective", "lambdarank")
        prm = {"objective": obj, "metric": "ndcg" if obj == "lambdarank" else "None",
               "ndcg_eval_at": [1], "label_gain": label_gain_for(grades),
               "verbosity": -1, "seed": 0, "num_threads": int(os.environ.get("LGBM_THREADS", 8)),
               "bagging_freq": 1, **p}
        if obj == "lambdarank":
            lab = lambda d: graded_labels(d["finish_pos"].to_numpy(), grades)
        else:   # rank_xendcg: 勝ち馬だけを正例にすると softmax の交差エントロピーに近い
            lab = lambda d: np.clip(grades + 1 - d["finish_pos"].to_numpy(), 0, grades).astype(int)
        dtr = lgb.Dataset(fit_df[cols].to_numpy(np.float32), label=lab(fit_df), group=group_sizes(fit_df),
                          free_raw_data=True)
        dcal = lgb.Dataset(cal_df[cols].to_numpy(np.float32), label=lab(cal_df), group=group_sizes(cal_df),
                           reference=dtr)
        rid_cal = cal_df["race_id"].to_numpy()
        y_cal = cal_df["is_win"].to_numpy()

        def feval(preds, _ds):
            return "race_nll", metrics.race_nll(metrics.race_softmax(preds, rid_cal), y_cal, rid_cal), False

        bst = lgb.train(prm, dtr, num_boost_round=rounds, valid_sets=[dcal], feval=feval,
                        callbacks=[lgb.early_stopping(100, first_metric_only=False, verbose=False)])
        info["best_iteration"] = bst.best_iteration
        if KEEP_MODEL:
            info["model"] = bst
        outs = [metrics.race_softmax(bst.predict(t[cols].to_numpy(np.float32), num_iteration=bst.best_iteration),
                                     t["race_id"].to_numpy()) for t in targets]
        return outs, info

    if model == "catboost":
        from nar.models.catboost_rank import CatBoostRanker

        prm = {"iterations": int(p.pop("_max_rounds", 5000)), "task_type": os.environ.get("CB_TASK", "GPU"),
               **p}
        if prm["task_type"] == "CPU" and os.environ.get("CB_THREADS"):
            prm["thread_count"] = int(os.environ["CB_THREADS"])
        if prm["task_type"] == "GPU":
            prm.pop("rsm", None)                       # GPU は rsm（列サンプリング）非対応（ランキング損失）
            # TabM と GPU を分け合うため VRAM の上限を指定できる（既定 0.95）
            if os.environ.get("CB_GPU_RAM_PART"):
                prm["gpu_ram_part"] = float(os.environ["CB_GPU_RAM_PART"])
        m = CatBoostRanker(prm).fit(fit_df, cols, eval_df=cal_df)
        info["best_iteration"] = m.best_iteration
        if KEEP_MODEL:
            info["model"] = m
        return [m.predict_proba(t) for t in targets], info

    if model == "tabm":
        from nar.models.base import to_batch
        from nar.models.tabm import TabM, TabMConfig

        cfg = TabMConfig(**{k: v for k, v in p.items() if k in TabMConfig.__annotations__})
        m = TabM(cfg).fit(to_batch(fit_df, cols), to_batch(cal_df, cols))
        outs = []
        for t in targets:
            b = to_batch(t, cols)
            outs.append(b.flat_predictions(m.predict_proba(b), len(t)))
        if KEEP_MODEL:
            info["model"] = m
            return outs, info
        del m
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return outs, info

    if model == "bayes":
        from nar.models.bayes import BayesConfig, HierarchicalPlackettLuce, build_arrays, restrict_likelihood_to

        n_races = int(p.get("max_races", 4000))
        tr = pd.concat([fit_df, cal_df])
        recent = (tr[["race_id", "start_ts"]].drop_duplicates().sort_values("start_ts")
                  .tail(n_races)["race_id"])
        tr = tr[tr["race_id"].isin(set(recent))]
        combined = pd.concat([tr, *targets], ignore_index=True)
        arr = build_arrays(combined, cols, max_positions=int(p.get("max_positions", 3)))
        fit_arr = restrict_likelihood_to(arr, set(tr["race_id"]))
        bcfg = BayesConfig(svi_steps=int(p.get("svi_steps", 3000)), svi_lr=float(p.get("svi_lr", 5e-3)),
                           max_positions=int(p.get("max_positions", 3)))
        model_ = HierarchicalPlackettLuce(bcfg).fit_svi(fit_arr)
        p_all = model_.flat_predictions(arr, len(combined))
        outs, off = [], len(tr)
        for t in targets:
            outs.append(p_all[off:off + len(t)])
            off += len(t)
        return outs, info
    raise ValueError(model)


def finish_probs(p: np.ndarray, rid: np.ndarray) -> np.ndarray:
    return metrics.normalize_within_race(np.clip(np.nan_to_num(p, nan=1e-9), 1e-12, None), rid)


def calibrated(p_cal, cal_df, p_list, t_list):
    """較正区間で温度を合わせ、各 target に適用。"""
    rid = cal_df["race_id"].to_numpy()
    sc = TemperatureScaler().fit(np.log(finish_probs(p_cal, rid)), cal_df["is_win"].to_numpy(), rid)
    outs = [sc.transform(np.log(finish_probs(p, t["race_id"].to_numpy())), t["race_id"].to_numpy())
            for p, t in zip(p_list, t_list)]
    return outs, float(sc.temperature)


# ====================================================================== 特徴量の順位
def cmd_rank(a) -> int:
    from nar.features.selection import drop_collinear, null_importance

    df, cols = load(a.variant)
    d = window(df, f"{RANK_YEARS[0]}-01-01", f"{RANK_YEARS[1]}-12-31")
    stats = fit_stats(d, cols)
    d = standardize(d, cols, stats)
    t0 = time.time()
    ni = null_importance(d, cols, n_runs=a.null_runs, seed=0, num_boost_round=150)
    ni["score"] = ni["actual_gain"] - ni["null_p95"]
    ni["passed"] = ni["actual_gain"] > ni["null_p95"]
    passed = ni[ni["passed"]].sort_values("score", ascending=False)["feature"].tolist()
    kept, dropped = drop_collinear(d, passed)
    rest = ni[~ni["feature"].isin(kept)].sort_values("actual_gain", ascending=False)["feature"].tolist()
    ranking = list(kept) + rest
    out = OUT / a.variant
    out.mkdir(parents=True, exist_ok=True)
    ni.to_csv(out / "null_importance.csv", index=False)
    (out / "feature_ranking.json").write_text(json.dumps(
        {"ranking": ranking, "n_passed": len(kept), "dropped_collinear": list(dropped),
         "period": RANK_YEARS, "seconds": time.time() - t0}, ensure_ascii=False, indent=1))
    log.info("%s: 通過 %d / 全 %d（共線で除外 %d）", a.variant, len(kept), len(cols), len(dropped))
    return 0


def ranking(variant: str) -> tuple[list[str], int]:
    r = json.loads((OUT / variant / "feature_ranking.json").read_text())
    return r["ranking"], r["n_passed"]


N_FEAT_GRID = (8, 15, 25, 35, 50, 70, 999)


# ====================================================================== HPO
def suggest(model: str, t, n_all: int) -> dict:
    n_opts = sorted({min(n, n_all) for n in N_FEAT_GRID})
    p: dict = {"_n_features": t.suggest_categorical("_n_features", n_opts)}
    if model != "bayes":
        p["_train_years"] = t.suggest_categorical("_train_years", [4, 8])
    if model == "clogit":
        p["l2"] = t.suggest_float("l2", 1e-8, 1.0, log=True)
        p["l1"] = t.suggest_float("l1", 1e-8, 1e-1, log=True)
    elif model == "lgbm":
        p.update({
            "objective": t.suggest_categorical("objective", ["lambdarank", "rank_xendcg"]),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": t.suggest_int("num_leaves", 7, 255, log=True),
            "min_data_in_leaf": t.suggest_int("min_data_in_leaf", 20, 2000, log=True),
            "feature_fraction": t.suggest_float("feature_fraction", 0.3, 1.0),
            "bagging_fraction": t.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": t.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": t.suggest_float("lambda_l2", 1e-8, 100.0, log=True),
            "min_gain_to_split": t.suggest_float("min_gain_to_split", 1e-8, 1.0, log=True),
            "_label_grades": t.suggest_categorical("_label_grades", [3, 5]),
        })
    elif model == "catboost":
        p.update({
            "loss_function": t.suggest_categorical("loss_function", ["QuerySoftMax", "YetiRank"]),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": t.suggest_int("depth", 4, 10),
            "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1e-2, 100.0, log=True),
            "random_strength": t.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "bagging_temperature": t.suggest_float("bagging_temperature", 0.0, 2.0),
            "border_count": t.suggest_categorical("border_count", [64, 128, 254]),
        })
    elif model == "tabm":
        p.update({
            "k": t.suggest_categorical("k", [4, 8, 16]),
            "hidden": t.suggest_categorical("hidden", [128, 256, 512]),
            "n_layers": t.suggest_int("n_layers", 2, 4),
            "dropout": t.suggest_float("dropout", 0.0, 0.5),
            "lr": t.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": t.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "epochs": t.suggest_int("epochs", 5, 40),
            "batch_races": t.suggest_categorical("batch_races", [128, 256, 512]),
        })
    elif model == "bayes":
        p.update({
            "max_races": t.suggest_categorical("max_races", [2000, 4000, 8000, 16000]),
            "svi_steps": t.suggest_categorical("svi_steps", [2000, 4000, 8000]),
            "svi_lr": t.suggest_float("svi_lr", 1e-3, 3e-2, log=True),
            "max_positions": t.suggest_int("max_positions", 1, 4),
        })
    return p


def model_params(p: dict) -> dict:
    return {k: v for k, v in p.items() if k not in ("_n_features", "_train_years")}


def evaluate_config(model: str, params: dict, variant: str, splits_cache: dict, trial=None) -> float:
    df, _ = load(variant)
    rank_list, _ = ranking(variant)
    n = params["_n_features"]
    cols = rank_list[:n] if model != "bayes" else rank_list[:min(n, 10)]
    years = params.get("_train_years", 4)
    scores = []
    for i, vy in enumerate(TUNE_VALID_YEARS):
        vs = f"{vy}-02-01"
        key = (vy, years, tuple(cols))
        if key not in splits_cache:
            splits_cache.clear()                     # メモリ節約（同じ設定の連続呼び出しだけ使い回す）
        fit_raw, cal_raw = split_train(df, vs, years)
        va_raw = window(df, vs, f"{vy}-12-31")
        stats = fit_stats(fit_raw, cols)
        fit_df, cal_df, va = (standardize(x, cols, stats) for x in (fit_raw, cal_raw, va_raw))
        (p_cal, p_va), _ = fit_predict(model, model_params(params), fit_df, cal_df, [cal_df, va], cols)
        (p_va_c,), _t = calibrated(p_cal, cal_df, [p_va], [va])
        s = metrics.race_nll(p_va_c, va["is_win"].to_numpy(), va["race_id"].to_numpy())
        scores.append(s)
        del fit_df, cal_df, va, p_cal, p_va
        gc.collect()
        if trial is not None:
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
    return float(np.mean(scores))


def cmd_hpo(a) -> int:
    import optuna

    df, cols = load(a.variant)
    rank_list, _ = ranking(a.variant)
    out = OUT / a.variant
    out.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{out / 'hpo.sqlite'}"
    study = optuna.create_study(
        study_name=f"{a.variant}_{a.model}", storage=storage, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=hash((a.variant, a.model)) % 10_000, multivariate=True,
                                           n_startup_trials=min(20, max(5, a.trials // 10))),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))
    done = len([t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE,
                                                       optuna.trial.TrialState.PRUNED)])
    left = max(0, a.trials - done)
    log.info("%s/%s: 済み %d 試行、残り %d", a.variant, a.model, done, left)
    cache: dict = {}

    def objective(t):
        p = suggest(a.model, t, len(rank_list))
        # 2026-10-16: hidden=512 は計算量が 4 倍で、上位試行に1つも無い。1 試行 5 時間超になり
        # 予定（10/22 完了）に収まらないため学習せずに打ち切る
        if a.model == "tabm" and p.get("hidden") == 512:
            raise optuna.TrialPruned()
        t0 = time.time()
        v = evaluate_config(a.model, p, a.variant, cache, trial=t)
        t.set_user_attr("seconds", time.time() - t0)
        return v

    def stop_at_limit(st, _trial):
        # 試行数の上限は artifacts/all_models/trial_limits.json で途中から変えられる
        # （2026-10-15: 設計書の予算では 3〜4 週間かかるため、ユーザー判断で削減）
        lim = trial_limit(a.model, a.trials, a.variant)
        n = len([t for t in st.trials if t.state in (optuna.trial.TrialState.COMPLETE,
                                                     optuna.trial.TrialState.PRUNED)])
        if n >= lim:
            log.info("%s/%s: 上限 %d 試行に達したので終了", a.variant, a.model, lim)
            st.stop()

    left = max(0, trial_limit(a.model, a.trials, a.variant) - done)
    if left:
        study.optimize(objective, n_trials=left, catch=(Exception,), gc_after_trial=True,
                       callbacks=[stop_at_limit])
    best = study.best_trial
    (out / f"best_{a.model}.json").write_text(json.dumps(
        {"params": best.params, "value": best.value, "n_trials": len(study.trials)}, ensure_ascii=False, indent=1))
    log.info("%s/%s 最良 %.5f %s", a.variant, a.model, best.value, best.params)
    return 0


def trial_limit(model: str, default: int, variant: str | None = None) -> int:
    """上限。"variant/model" の指定があればそれ、なければ model の指定、なければ default。"""
    f = OUT / "trial_limits.json"
    if f.exists():
        d = json.loads(f.read_text())
        return int(d.get(f"{variant}/{model}", d.get(model, default)))
    return default


def deferred(variant: str, model: str) -> bool:
    """artifacts/all_models/defer/<variant>_<model> があれば fit を後回しにする（メモリの都合）。"""
    return (OUT / "defer" / f"{variant}_{model}").exists()


def best_params(variant: str, model: str) -> dict:
    b = json.loads((OUT / variant / f"best_{model}.json").read_text())["params"]
    return b


# ====================================================================== walk-forward
def run_period(model: str, variant: str, params: dict, valid_start: str, valid_end: str | None):
    df, _ = load(variant)
    rank_list, _ = ranking(variant)
    n = int(params["_n_features"])
    cols = rank_list[:n] if model != "bayes" else rank_list[:min(n, 10)]
    years = params.get("_train_years", 4)
    fit_raw, cal_raw = split_train(df, valid_start, years)
    va_raw = window(df, valid_start, valid_end or "2100-01-01")
    stats = fit_stats(fit_raw, cols)
    fit_df, cal_df, va = (standardize(x, cols, stats) for x in (fit_raw, cal_raw, va_raw))
    t0 = time.time()
    (p_cal, p_va), info = fit_predict(model, model_params(params), fit_df, cal_df, [cal_df, va], cols)
    (p_va_c,), temp = calibrated(p_cal, cal_df, [p_va], [va])
    out = va_raw[["race_id", "horse_no", "race_date", "is_win", "finish_pos", "dnf"]].copy()
    out[model] = p_va_c
    m = metrics.summary(p_va_c, va["is_win"].to_numpy(), va["finish_pos"].to_numpy(), va["race_id"].to_numpy())
    return out, {**m, **info, "temperature": temp, "seconds": time.time() - t0, "n_fit": len(fit_df)}


def cmd_fit(a) -> int:
    if deferred(a.variant, a.model):
        log.info("%s/%s: fit は後回し（%s）", a.variant, a.model, OUT / "defer")
        return 0
    params = best_params(a.variant, a.model)
    out = OUT / a.variant / "oof"
    out.mkdir(parents=True, exist_ok=True)
    for k, (vs, ve) in FOLDS.items():
        path = out / f"{a.model}_fold{k}.parquet"
        if path.exists():
            continue
        pred, m = run_period(a.model, a.variant, params, vs, ve)
        pred.assign(fold=k).to_parquet(path, index=False)
        (out / f"{a.model}_fold{k}.json").write_text(json.dumps(m, default=float, indent=1))
        log.info("%s/%s fold%d: nll %.5f top1 %.4f (%.0fs)", a.variant, a.model, k,
                 m["race_nll"], m["top1"], m["seconds"])
        gc.collect()
    return 0


def oof(variant: str, models=MODELS, period: str = "oof") -> pd.DataFrame:
    base = None
    d = OUT / variant / period
    for m in models:
        files = sorted(d.glob(f"{m}_fold*.parquet")) if period == "oof" else sorted(d.glob(f"{m}.parquet"))
        if not files:
            continue
        f = pd.concat([pd.read_parquet(x) for x in files], ignore_index=True)
        if base is None:
            base = f
        else:
            base = base.merge(f[["race_id", "horse_no", m]], on=["race_id", "horse_no"], how="inner")
    return base


def cmd_ensemble(a) -> int:
    from nar.models.ensemble import ConstrainedStacker, log_average

    o = oof(a.variant)
    models = [m for m in MODELS if m in o.columns]
    rows = []
    o = o.sort_values(["race_id", "horse_no"]).reset_index(drop=True)
    ens = pd.Series(np.nan, index=o.index)
    weights = {}
    for k in sorted(o["fold"].unique()):
        past = o[o["fold"] < k]
        cur = o["fold"] == k
        if past.empty:
            continue
        st = ConstrainedStacker(models).fit(past[models], past["is_win"].to_numpy(), past["race_id"].to_numpy())
        weights[int(k)] = dict(zip(models, map(float, st.weights)))
        ens[cur] = st.predict_proba(o.loc[cur, models], o.loc[cur, "race_id"].to_numpy())
    o["ensemble_wf"] = ens
    o["log_avg"] = log_average(o[models], o["race_id"].to_numpy())
    st_all = ConstrainedStacker(models).fit(o[models], o["is_win"].to_numpy(), o["race_id"].to_numpy())
    for k, g in o.groupby("fold"):
        for m in [*models, "log_avg", "ensemble_wf"]:
            if g[m].isna().any():
                continue
            rows.append({"fold": int(k), "model": m, **metrics.summary(
                g[m].to_numpy(), g["is_win"].to_numpy(), g["finish_pos"].to_numpy(), g["race_id"].to_numpy())})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / a.variant / "fold_metrics.csv", index=False)
    o.to_parquet(OUT / a.variant / "oof_all.parquet", index=False)
    (OUT / a.variant / "ensemble.json").write_text(json.dumps(
        {"walk_forward_weights": weights, "all_fold_weights": dict(zip(models, map(float, st_all.weights)))},
        indent=1))
    pd.set_option("display.width", 200)
    print(r.pivot_table(index="model", columns="fold", values="race_nll").round(5))
    print(json.dumps(dict(zip(models, map(float, st_all.weights))), indent=1))
    return 0


# ====================================================================== OOS
def cmd_oos(a) -> int:
    from nar.config import cv_config
    from nar.eval.splits import OOSGuard

    ccfg = cv_config(str(CONF))
    guard = OOSGuard(ccfg, OUT / "oos_access.log")
    guard.unlock(a.reason)
    out = OUT / a.variant / "oos"
    out.mkdir(parents=True, exist_ok=True)
    for m in (a.models.split(",") if a.models else MODELS):
        path = out / f"{m}.parquet"
        if path.exists() or not (OUT / a.variant / f"best_{m}.json").exists():
            continue
        pred, met = run_period(m, a.variant, best_params(a.variant, m), OOS[0], OOS[1])
        pred.to_parquet(path, index=False)
        (out / f"{m}.json").write_text(json.dumps(met, default=float, indent=1))
        log.info("%s/%s OOS: nll %.5f top1 %.4f", a.variant, m, met["race_nll"], met["top1"])
    return 0


# ====================================================================== 計測
def cmd_bench(a) -> int:
    """1 設定 × 内側 1 split の所要時間（試行予算の見積もり用）。"""
    df, cols = load(a.variant)
    rank_list, _ = ranking(a.variant)
    defaults = {"clogit": {"l2": 1e-3, "l1": 1e-6}, "lgbm": {"objective": a.lgbm_obj, "learning_rate": 0.05},
                "catboost": {"loss_function": "QuerySoftMax", "learning_rate": 0.08, "depth": 6},
                "tabm": {"k": 8, "hidden": 256, "n_layers": 3, "epochs": 10},
                "bayes": {"max_races": 4000, "svi_steps": 3000}}
    for m in a.models.split(","):
        p = {"_n_features": 35, "_train_years": 4, **defaults[m]}
        cols_ = rank_list[:35] if m != "bayes" else rank_list[:6]
        fit_raw, cal_raw = split_train(df, "2018-02-01", 4)
        va_raw = window(df, "2018-02-01", "2018-12-31")
        stats = fit_stats(fit_raw, cols_)
        fit_df, cal_df, va = (standardize(x, cols_, stats) for x in (fit_raw, cal_raw, va_raw))
        t0 = time.time()
        (pc, pv), info = fit_predict(m, model_params(p), fit_df, cal_df, [cal_df, va], cols_)
        (pvc,), temp = calibrated(pc, cal_df, [pv], [va])
        nll = metrics.race_nll(pvc, va["is_win"].to_numpy(), va["race_id"].to_numpy())
        print(f"{m}: {time.time() - t0:.0f}s nll={nll:.5f} temp={temp:.3f} {info} rows={len(fit_df):,}", flush=True)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("rank"); r.add_argument("variant"); r.add_argument("--null-runs", type=int, default=30)
    h = sp.add_parser("hpo"); h.add_argument("variant"); h.add_argument("model"); h.add_argument("--trials", type=int, required=True)
    f = sp.add_parser("fit"); f.add_argument("variant"); f.add_argument("model")
    e = sp.add_parser("ensemble"); e.add_argument("variant")
    o = sp.add_parser("oos"); o.add_argument("variant"); o.add_argument("--reason", required=True); o.add_argument("--models", default="")
    b = sp.add_parser("bench"); b.add_argument("variant"); b.add_argument("--models", default=",".join(MODELS))
    b.add_argument("--lgbm-obj", default="lambdarank")
    a = ap.parse_args()
    return {"rank": cmd_rank, "hpo": cmd_hpo, "fit": cmd_fit, "ensemble": cmd_ensemble,
            "oos": cmd_oos, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
