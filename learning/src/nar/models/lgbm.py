"""LightGBM LambdaRank。

競馬はレース単位のグルーピングが本質なので二値分類ではなく lambdarank を使う。
ラベルは 1着=3 / 2着=2 / 3着=1 / その他=0 の段階付けで、勝ち馬だけでなく
上位入着の情報も学習に使う。

出力はスコアであって確率ではない。必ずレース内 softmax → 温度スケーリングを通す。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..eval.metrics import race_softmax


def graded_labels(finish_pos: np.ndarray) -> np.ndarray:
    """{0,1,2,3}。同着は同じ着順値を持つので同じラベルになる（MD-09 の例外処理）。

    着順が NULL の行（取消・除外で走っていない馬）は受け取らない。黙って 0 に
    倒すと「走っていない馬 = 4着以下」として学習してしまう。float の NaN を
    int にキャストすると int64 の最小値になり、LightGBM が
    「label should be int type (met -9223372036854775808)」で落ちる。
    落ちること自体は正しいので、原因が分かるところで止める。
    """
    pos = np.asarray(finish_pos, dtype=float)
    if np.isnan(pos).any():
        raise ValueError(
            f"着順が NULL の行が {int(np.isnan(pos).sum())} 件あります。"
            "pipeline.trainable() を通して未出走の馬を除いてください。")
    return np.clip(4 - pos, 0, 3).astype(int)


def group_sizes(df: pd.DataFrame, race_col: str = "race_id") -> np.ndarray:
    """行順を保ったままのレース頭数。

    LightGBM の group は「行が既にレース順に並んでいる」前提なので、
    並べ替えではなく連続性の検証を行う（MD-08）。
    """
    codes = pd.factorize(df[race_col], sort=False)[0]
    boundaries = np.flatnonzero(np.diff(codes)) + 1
    sizes = np.diff(np.concatenate([[0], boundaries, [len(codes)]]))
    if len(np.unique(codes)) != len(sizes):
        raise ValueError(
            "race_id が行方向で分断されています。group を渡す前にレース単位で連続させてください。"
        )
    return sizes


class LgbmRanker:
    def __init__(self, params: dict | None = None, num_boost_round: int = 300) -> None:
        self.params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3],
            "label_gain": [0, 1, 3, 7],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": 0,
            "deterministic": True,
            "force_row_wise": True,
            **(params or {}),
        }
        self.num_boost_round = num_boost_round
        self.booster = None
        self.feature_names: list[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: list[str],
            pos_col: str = "finish_pos", race_col: str = "race_id") -> "LgbmRanker":
        import lightgbm as lgb

        self.feature_names = list(feature_cols)
        groups = group_sizes(df, race_col)
        if groups.sum() != len(df):
            raise ValueError("group の総和が学習行数と一致しません（MD-07）")
        ds = lgb.Dataset(
            df[feature_cols], label=graded_labels(df[pos_col]), group=groups,
            feature_name=self.feature_names, free_raw_data=False,
        )
        self.booster = lgb.train(self.params, ds, num_boost_round=self.num_boost_round)
        return self

    def predict_proba(self, df: pd.DataFrame, race_col: str = "race_id") -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("fit() を先に呼んでください")
        scores = self.booster.predict(df[self.feature_names])
        return race_softmax(np.asarray(scores), df[race_col].to_numpy())

    def importance(self) -> pd.Series:
        return pd.Series(
            self.booster.feature_importance("gain"), index=self.feature_names
        ).sort_values(ascending=False)
