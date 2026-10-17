"""CatBoost のレース内ランキング（2026-10-13）。

損失は QuerySoftMax（グループ = レース）を既定にする。これはレース内 softmax の
交差エントロピー、つまり条件付きロジットと同じ尤度を勾配ブースティングで最適化する
もので、勝率の推定に一番素直。YetiRank（ペア/リストのランキング損失）も HPO の候補に入れる。

出力はスコア。lgbm と同じくレース内 softmax → 温度スケーリングを通す。
GPU（task_type="GPU"）は QuerySoftMax / YetiRank の両方に対応している。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..eval.metrics import race_softmax


class CatBoostRanker:
    def __init__(self, params: dict | None = None) -> None:
        p = dict(params or {})
        self.loss = p.pop("loss_function", "QuerySoftMax")
        self.params = {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": 0,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
            **p,
        }
        self.model = None
        self.feature_names: list[str] = []

    def _pool(self, df: pd.DataFrame, cols: list[str], race_col: str, with_label: bool):
        from catboost import Pool

        gid = pd.factorize(df[race_col], sort=False)[0]
        if (np.diff(gid) < 0).any():
            raise ValueError("race_id が行方向で分断されています（レース単位で連続させてください）")
        label = None
        if with_label:
            # QuerySoftMax は勝ち馬=1（レース内で和が 1 の目標）。YetiRank は大きいほど上位
            label = (df["is_win"].to_numpy(float) if self.loss.startswith("QuerySoftMax")
                     else np.clip(4 - df["finish_pos"].to_numpy(float), 0, 3))
        return Pool(df[cols].to_numpy(np.float32), label=label, group_id=gid)

    def fit(self, df: pd.DataFrame, feature_cols: list[str], race_col: str = "race_id",
            eval_df: pd.DataFrame | None = None) -> "CatBoostRanker":
        from catboost import CatBoost

        self.feature_names = list(feature_cols)
        params = {**self.params, "loss_function": self.loss}
        if eval_df is not None:
            params.setdefault("od_type", "Iter")
            params.setdefault("od_wait", 100)
            params["use_best_model"] = True
        self.model = CatBoost(params)
        self.model.fit(self._pool(df, feature_cols, race_col, True),
                       eval_set=(self._pool(eval_df, feature_cols, race_col, True)
                                 if eval_df is not None else None))
        return self

    @property
    def best_iteration(self) -> int | None:
        return None if self.model is None else self.model.get_best_iteration()

    def predict_proba(self, df: pd.DataFrame, race_col: str = "race_id") -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() を先に呼んでください")
        s = self.model.predict(self._pool(df, self.feature_names, race_col, False))
        return race_softmax(np.asarray(s, dtype=float), df[race_col].to_numpy())

    def importance(self) -> pd.Series:
        return pd.Series(self.model.get_feature_importance(), index=self.feature_names
                         ).sort_values(ascending=False)
