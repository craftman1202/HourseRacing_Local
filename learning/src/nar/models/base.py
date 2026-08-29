"""レース単位のデータ構造。

頭数が可変なのでパディング＋マスクで扱う。パディング要素は出力に一切影響しない
ことが不変条件（MD-05 / MD-12）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RaceBatch:
    x: np.ndarray          # (n_races, max_n, n_features)
    y: np.ndarray          # (n_races, max_n) 勝ち馬 1
    mask: np.ndarray       # (n_races, max_n) 有効馬 True
    race_ids: np.ndarray   # (n_races,)
    row_index: np.ndarray  # (n_races, max_n) 元 DataFrame の行位置。-1 はパディング
    feature_names: tuple[str, ...]

    @property
    def n_races(self) -> int:
        return self.x.shape[0]

    def flat_predictions(self, scores: np.ndarray, n_rows: int) -> np.ndarray:
        """(n_races, max_n) を元の行順に戻す。"""
        out = np.full(n_rows, np.nan)
        valid = self.row_index >= 0
        out[self.row_index[valid]] = scores[valid]
        return out


def to_batch(
    df: pd.DataFrame, feature_cols: list[str],
    race_col: str = "race_id", label_col: str = "is_win",
    pad_value: float = 0.0,
) -> RaceBatch:
    groups = list(df.groupby(race_col, sort=False).indices.items())
    max_n = max(len(idx) for _, idx in groups)
    n_races, n_feat = len(groups), len(feature_cols)

    x = np.full((n_races, max_n, n_feat), pad_value, dtype=float)
    y = np.zeros((n_races, max_n), dtype=float)
    mask = np.zeros((n_races, max_n), dtype=bool)
    row_index = np.full((n_races, max_n), -1, dtype=int)
    race_ids = np.empty(n_races, dtype=object)

    values = df[feature_cols].to_numpy(dtype=float)
    labels = df[label_col].to_numpy(dtype=float)
    for i, (rid, idx) in enumerate(groups):
        n = len(idx)
        x[i, :n] = values[idx]
        y[i, :n] = labels[idx]
        mask[i, :n] = True
        row_index[i, :n] = idx
        race_ids[i] = rid
    return RaceBatch(x, y, mask, race_ids, row_index, tuple(feature_cols))


def masked_softmax(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """パディング位置に確率を漏らさない softmax。"""
    s = np.where(mask, scores, -np.inf)
    s = s - np.max(s, axis=1, keepdims=True)
    e = np.where(mask, np.exp(s), 0.0)
    return e / e.sum(axis=1, keepdims=True)
