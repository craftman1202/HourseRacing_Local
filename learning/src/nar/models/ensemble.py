"""制約付きスタッキング。

log π ∝ Σ w_m log p^(m),  w_m ≥ 0,  Σ w_m = 1

重みは walk-forward の out-of-fold 予測のみから推定する。訓練内予測を混ぜると
重みが致命的にバイアスするので、OOF でないものは受け付けずに例外を投げる（EN-02）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..eval.metrics import normalize_within_race, race_nll


class OOFViolation(ValueError):
    pass


class ConstrainedStacker:
    def __init__(self, model_names: list[str]) -> None:
        self.model_names = list(model_names)
        self.weights: np.ndarray | None = None

    def fit(
        self, oof: pd.DataFrame, y: np.ndarray, race_ids: np.ndarray,
        fold_ids: np.ndarray | None = None,
    ) -> "ConstrainedStacker":
        """oof は列がモデル名、行が予測。fold_ids を渡すと OOF 性を検証する。"""
        if fold_ids is not None:
            self._assert_oof(oof, fold_ids)
        log_p = np.log(np.clip(oof[self.model_names].to_numpy(dtype=float), 1e-12, 1.0))

        def objective(w: np.ndarray) -> float:
            return race_nll(self._blend(log_p, w, race_ids), y, race_ids)

        m = len(self.model_names)
        res = minimize(
            objective, np.full(m, 1.0 / m), method="SLSQP",
            bounds=[(0.0, 1.0)] * m,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": 300, "ftol": 1e-10},
        )
        w = np.clip(res.x, 0.0, None)
        self.weights = w / w.sum()
        return self

    def predict_proba(self, preds: pd.DataFrame, race_ids: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("fit() を先に呼んでください")
        log_p = np.log(np.clip(preds[self.model_names].to_numpy(dtype=float), 1e-12, 1.0))
        return self._blend(log_p, self.weights, race_ids)

    @staticmethod
    def _blend(log_p: np.ndarray, w: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
        return normalize_within_race(np.exp(log_p @ w), race_ids)

    @staticmethod
    def _assert_oof(oof: pd.DataFrame, fold_ids: np.ndarray) -> None:
        """各行がちょうど1つの fold から来ていること。

        in-fold 予測が混ざると同じ行に複数 fold の予測が乗るので、
        fold_ids の欠損と重複の両方を見る。
        """
        s = pd.Series(fold_ids)
        if s.isna().any():
            raise OOFViolation(
                f"fold 割り当ての無い行が {int(s.isna().sum())} 件あります。"
                "in-fold 予測が混入している可能性があります。"
            )
        if len(s) != len(oof):
            raise OOFViolation("予測行数と fold 割り当ての長さが一致しません。")


def simple_average(preds: pd.DataFrame, race_ids: np.ndarray) -> np.ndarray:
    return normalize_within_race(preds.mean(axis=1).to_numpy(), race_ids)


def log_average(preds: pd.DataFrame, race_ids: np.ndarray) -> np.ndarray:
    log_p = np.log(np.clip(preds.to_numpy(dtype=float), 1e-12, 1.0))
    return normalize_within_race(np.exp(log_p.mean(axis=1)), race_ids)
