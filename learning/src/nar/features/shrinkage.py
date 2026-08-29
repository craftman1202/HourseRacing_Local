"""経験ベイズ収縮。

騎手・種牡馬は出走数が極端に偏るので、生の勝率を特徴量にすると
「1戦1勝＝勝率100%」が最強の馬になる。α は HPO の探索対象。
"""

from __future__ import annotations

import numpy as np


def shrink(wins: np.ndarray, starts: np.ndarray, prior: float, alpha: float) -> np.ndarray:
    """(wins + α·p̄) / (starts + α)。

    starts=0 で p̄、starts→∞ で生の勝率に収束する（FE-05）。
    """
    if alpha <= 0:
        raise ValueError("alpha は正である必要があります（0 だと収縮が無効化されます）")
    wins = np.asarray(wins, dtype=float)
    starts = np.asarray(starts, dtype=float)
    return (wins + alpha * prior) / (starts + alpha)


def sql_shrink(wins_expr: str, starts_expr: str, prior_expr: str, alpha: float) -> str:
    """同じ式を DuckDB 側で書くための文字列。Python 側と定義がズレないように一本化する。"""
    return f"(({wins_expr}) + {alpha} * ({prior_expr})) / (({starts_expr}) + {alpha})"
