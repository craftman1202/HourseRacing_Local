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


# ラベルの段階数。設計書 §8.2 / §13.1-3。
# 3 = 1着3 / 2着2 / 3着1 / その他0（従来の既定）
# 5 = 1着5 …… 5着1 / 6着以下0（Research.md §2.2 の提案）
# どちらが良いかは頭数分布に依存するので固定せず HPO で選ばせる。NAR は少頭数
# 開催が多く、段階を増やすと下位グレードが「実質全馬」になって信号が薄まりうる。
LABEL_GRADES = (3, 5)


def label_gain_for(grades: int) -> list[float]:
    """LightGBM の label_gain。2^rel − 1（NDCG の既定と同じ形）。

    長さは最大ラベル+1 でなければならず、足りないと LightGBM が
    「label_gain has 4 elements, but max label is 5」で落ちる。
    """
    return [float(2 ** i - 1) for i in range(grades + 1)]


def graded_labels(finish_pos: np.ndarray, grades: int = 3) -> np.ndarray:
    """{0..grades}。同着は同じ着順値を持つので同じラベルになる（MD-09 の例外処理）。

    着順が NULL の行（取消・除外で走っていない馬）は受け取らない。黙って 0 に
    倒すと「走っていない馬 = 4着以下」として学習してしまう。float の NaN を
    int にキャストすると int64 の最小値になり、LightGBM が
    「label should be int type (met -9223372036854775808)」で落ちる。
    落ちること自体は正しいので、原因が分かるところで止める。
    """
    if grades not in LABEL_GRADES:
        raise ValueError(f"grades は {LABEL_GRADES} のいずれかにしてください（受領: {grades}）")
    pos = np.asarray(finish_pos, dtype=float)
    if np.isnan(pos).any():
        raise ValueError(
            f"着順が NULL の行が {int(np.isnan(pos).sum())} 件あります。"
            "pipeline.trainable() を通して未出走の馬を除いてください。")
    return np.clip(grades + 1 - pos, 0, grades).astype(int)


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
    def __init__(self, params: dict | None = None, num_boost_round: int = 300,
                 label_grades: int = 3) -> None:
        # `_label_grades` は HPO 側から params に混ぜて渡ってくる。LightGBM は
        # 知らないキーを警告付きで無視するだけなので、ここで確実に抜いておく。
        params = dict(params or {})
        label_grades = int(params.pop("_label_grades", label_grades))
        self.label_grades = label_grades
        self.params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3],
            "label_gain": label_gain_for(label_grades),
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
            **params,
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
            df[feature_cols], label=graded_labels(df[pos_col], self.label_grades),
            group=groups,
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
