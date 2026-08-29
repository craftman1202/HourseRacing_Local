"""比較の基準線。

トラックA がオッズベースラインを統計的に有意に上回るかどうかが、
このプロジェクト全体の成否を決める単一の問い（EV-03）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..eval.metrics import normalize_within_race


def uniform(df: pd.DataFrame, race_col: str = "race_id") -> np.ndarray:
    n = df.groupby(race_col)[race_col].transform("size")
    return (1.0 / n).to_numpy()


def market(df: pd.DataFrame, odds_col: str = "odds_win", race_col: str = "race_id",
           takeout: float = 0.20) -> np.ndarray:
    """市場暗黙確率。控除率で割り戻したうえでレース内総和を 1 に正規化する（EV-02）。

    正規化を省くと Σq = 1/(1-τ) ≈ 1.25 になり、NLL が不当に良く出る。
    """
    raw = (1.0 - takeout) / df[odds_col].to_numpy(dtype=float)
    return normalize_within_race(raw, df[race_col].to_numpy())


def favourite(df: pd.DataFrame, pop_col: str = "popularity", race_col: str = "race_id",
              mass: float = 0.999) -> np.ndarray:
    """1番人気固定。NLL が発散しないよう残余質量を配る。"""
    n = df.groupby(race_col)[race_col].transform("size").to_numpy()
    is_fav = (df[pop_col].to_numpy() == 1).astype(float)
    return is_fav * mass + (1 - is_fav) * (1 - mass) / np.maximum(n - 1, 1)
