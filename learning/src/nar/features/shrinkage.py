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


# Wilson スコア信頼区間の下限に使う z。1.96 = 両側95%。
WILSON_Z = 1.96


def wilson_lower(wins: np.ndarray, starts: np.ndarray, z: float = WILSON_Z) -> np.ndarray:
    """Wilson スコア信頼区間の下限。

    経験ベイズ収縮（shrink）が事前確率へ寄せるのに対し、こちらは**標本サイズの
    不確実性そのものをペナルティにする**。1戦1勝のペアは点推定 1.0 でも下限は
    0.21 にしかならず、「まだ実績が足りない組み合わせ」を低く評価できる。

    騎手×調教師のような組み合わせキーは出走数の分布が極端に歪む（大半が数戦、
    一部が数千戦）ので、収縮の事前確率を決め打ちするより不確実性で割り引くほうが
    素直。設計書 §13.1-1 / Research.md §3.1。

    starts=0 は 0.0 を返す（実績ゼロ＝下限ゼロ。定義上も自然）。
    """
    wins = np.asarray(wins, dtype=float)
    starts = np.asarray(starts, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.divide(wins, starts, out=np.zeros_like(wins), where=starts > 0)
        z2 = z * z
        denom = 1.0 + z2 / starts
        center = p + z2 / (2.0 * starts)
        margin = z * np.sqrt((p * (1.0 - p) + z2 / (4.0 * starts)) / starts)
        out = np.divide(center - margin, denom,
                        out=np.zeros_like(wins), where=starts > 0)
    return np.where(starts > 0, np.clip(out, 0.0, 1.0), 0.0)


def sql_wilson_lower(wins_expr: str, starts_expr: str, z: float = WILSON_Z) -> str:
    """wilson_lower と同じ式の DuckDB 版。定義の二重管理を避けるためここに置く。

    整数のウィンドウ和（COUNT / SUM(is_win)）から行ごとに閉形式で計算するだけなので、
    加算順に依存しない＝ FE-09（決定性）と LK-05（ビット単位一致）を壊さない。
    """
    n, w, z2 = f"({starts_expr})", f"({wins_expr})", z * z
    p = f"({w} / NULLIF({n}, 0))"
    return (
        f"CASE WHEN {n} > 0 THEN "
        f"(({p} + {z2} / (2.0 * {n})) - "
        f"{z} * SQRT(({p} * (1.0 - {p}) + {z2} / (4.0 * {n})) / {n})) "
        f"/ (1.0 + {z2} / {n}) "
        f"ELSE 0.0 END"
    )
