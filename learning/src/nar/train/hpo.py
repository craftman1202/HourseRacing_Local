"""nested HPO。

外側が walk-forward 5-fold、内側は各 fold の学習期間をさらに時系列分割した 3-fold。
目的関数は内側平均のレース内 NLL。

NLL を主指標にするのは、後段の期待値計算が確率の較正精度に直接依存するため。
Top-1 精度や NDCG で最適化しても較正は改善しない。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import optuna
import pandas as pd

from ..eval.metrics import race_nll
from ..models.lgbm import LABEL_GRADES
from ..eval.splits import Fold, inner_folds

optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger(__name__)

# 探索空間の次元数に応じた試行数。DL 側に十分な予算を与えないと
# GBDT 有利のバイアスがかかる（TabArena の知見）。
DEFAULT_TRIALS = {"lgbm": 500, "tabm": 300, "clogit": 100, "bayes": 50}


@dataclass
class HPOResult:
    model: str
    best_params: dict
    best_value: float
    n_trials: int
    trials: pd.DataFrame = field(default_factory=pd.DataFrame)


def suggest_lgbm(t: optuna.Trial) -> dict:
    return {
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": t.suggest_int("num_leaves", 15, 255, log=True),
        "min_data_in_leaf": t.suggest_int("min_data_in_leaf", 20, 500, log=True),
        "feature_fraction": t.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": t.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": t.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": t.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        # LambdaRank はレース内の全ペアに対して勾配を計算するため、ラウンド数が
        # そのまま実行時間になる（実測 200k 行で約 0.47 秒/ラウンド）。
        # 上限を伸ばすと nested HPO が現実的な時間で終わらないので、ここで抑える。
        "_num_boost_round": t.suggest_int("_num_boost_round", 100, 400, step=50),
        # ラベルの段階数（設計書 §13.1-3）。3段階（従来）と5段階（Research.md §2.2）を
        # 探索空間に入れ、どちらが良いかを NLL で決めさせる。決め打ちしない。
        "_label_grades": t.suggest_categorical("_label_grades", list(LABEL_GRADES)),
    }


def suggest_tabm(t: optuna.Trial) -> dict:
    """
    2026-09-17: 探索空間を `k∈{4,8,16}, hidden∈{128,256,512}, n_layers∈{2,3,4}`
    （設計値）から `k∈{4,8}, hidden∈{128,256}, n_layers∈{2,3}` へ狭めた。
    共有バックボーンのコストはおおよそ hidden² × n_layers × k に比例するため、
    設計値の探索空間では最悪ケース（k=16, hidden=512, n_layers=4）が最良ケース
    （k=4, hidden=128, n_layers=2）の理論上 100 倍超になりうる。CPU 実行かつ
    GPU が使えない期間（README「実行環境」節）に、単発の TPE サンプルがこの
    範囲を引くと nested HPO 1 fold が数時間〜1日規模に膨らみ、5 fold × 2 系統の
    現実的な実行時間を壊す。狭めた範囲でも最悪/最良の比は 12 倍程度に収まる。
    GPU 復旧後（WSL2 の GPU パススルーが直ったら）は設計値に戻す価値がある。
    """
    return {
        "k": t.suggest_categorical("k", [4, 8]),
        "hidden": t.suggest_categorical("hidden", [128, 256]),
        "n_layers": t.suggest_int("n_layers", 2, 3),
        "dropout": t.suggest_float("dropout", 0.0, 0.4),
        "lr": t.suggest_float("lr", 3e-4, 5e-3, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
    }


def suggest_clogit(t: optuna.Trial) -> dict:
    return {
        "l2": t.suggest_float("l2", 1e-8, 1.0, log=True),
        "l1": t.suggest_float("l1", 1e-8, 1.0, log=True),
    }


SUGGESTERS: dict[str, Callable[[optuna.Trial], dict]] = {
    "lgbm": suggest_lgbm, "tabm": suggest_tabm, "clogit": suggest_clogit,
}


def run(
    model: str,
    outer: Fold,
    feat: pd.DataFrame,
    feature_cols: list[str],
    train_predict: Callable[[dict, pd.DataFrame, pd.DataFrame, list[str]], np.ndarray],
    n_trials: int | None = None,
    embargo_days: int = 180,
    n_inner: int = 3,
    seed: int = 0,
    prepare_fn: Callable[[pd.DataFrame, list[str]], pd.DataFrame] | None = None,
) -> HPOResult:
    """内側 3-fold で params を探索する。

    train_predict(params, train_df, valid_df, cols) -> valid の予測確率。
    渡す train/valid は **prepare 済み**で、返す予測は valid の行順に一致していること。

    前処理をここで一度だけ済ませるのは、目的関数の整合性のため。以前は
    train_predict 側が内部で prepare（= race_id, horse_no でのソート）を行い、
    目的関数はソート前の va からラベルを取っていたため、予測とラベルが行単位で
    ずれ、HPO が実質シャッフルされたラベルに対して最適化していた。
    前処理の責務を1か所に集約してその経路自体を無くす。

    内側 fold は外側 train 期間内に完全に収まる（CV-09）。
    """
    inners = inner_folds(outer, n_inner, embargo_days)
    n_trials = n_trials or DEFAULT_TRIALS.get(model, 100)
    prep = prepare_fn or (lambda df, _cols: df)

    splits = []
    for f in inners:
        tr, va = f.mask(feat["race_date"])
        if tr.sum() < 100 or va.sum() < 100:
            continue
        splits.append((prep(feat[tr].copy(), feature_cols),
                       prep(feat[va].copy(), feature_cols)))
    if not splits:
        raise ValueError(
            f"{model}: 内側 fold に十分なデータがありません。"
            "外側 train 期間か embargo の設定を見直してください。")

    def objective(trial: optuna.Trial) -> float:
        params = SUGGESTERS[model](trial)
        scores = []
        for i, (tr, va) in enumerate(splits):
            p = np.asarray(train_predict(params, tr, va, feature_cols))
            if len(p) != len(va):
                raise ValueError(
                    f"{model}: 予測 {len(p)} 行が valid {len(va)} 行と一致しません。"
                    "train_predict は valid の行順どおりに返す必要があります。")
            scores.append(race_nll(p, va["is_win"].to_numpy(), va["race_id"].to_numpy()))
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        # fold ごとに探索列を変える。全 fold で同一シードだと TPE が同じ候補点を
        # 同じ順に評価し、best_params が全 fold で一致する（fold 内実行の意味が消える）。
        sampler=optuna.samplers.TPESampler(seed=seed + 1000 * outer.index),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    return HPOResult(model, dict(study.best_params), float(study.best_value),
                     len(study.trials), trials)
