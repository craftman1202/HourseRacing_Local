"""3連単「1番人気1着固定流し」（2026-10-06）。

検証（scripts/trifecta_cv.py、確率源 poponly・軸の単勝確率 ≥ 0.5）:
  5-fold（2019〜2023）平均 ROI 1.137（fold 最小 0.636）、OOS（2024-02〜）1.012 [0.84, 1.20]
  的中率 約 53%。選ばれるレースの 88% は 5〜6 頭立て（人気×頭数の勝率表で 1番人気が 50% に
  届くのは概ね 6 頭以下のときだけ）。

ルール: 人気だけの単勝確率（nar.eval.deviation.log_p_pop）で 1番人気 ≥ min_p なら、
  1着 = 1番人気、2・3着 = 人気 2〜(n_partners+1) 位の順序付き全組、1点 unit 円。
学習側の検証と運用側の通知が同じ関数を使う（SK-02）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

from .deviation import log_p_pop


@dataclass
class TrifectaPlan:
    axis: int                         # 1着に固定する馬番
    partners: list[int]               # 2・3着の候補（人気順）
    p_axis: float                     # 人気だけで見た軸の単勝確率
    tickets: list[tuple[int, int, int]] = field(default_factory=list)
    unit_yen: int = 100

    @property
    def stake_yen(self) -> int:
        return len(self.tickets) * self.unit_yen


def trifecta_fav_axis(horse_no: np.ndarray, popularity: np.ndarray, pop_table: dict,
                      min_p: float = 0.5, n_partners: int = 8,
                      unit_yen: int = 100) -> TrifectaPlan | None:
    """1レース分。人気が全頭そろっていない・3頭未満・条件未達なら None（買わない）。"""
    horse = np.asarray(horse_no, dtype=int)
    pop = np.asarray(popularity, dtype=float)
    n = len(horse)
    if n < 3 or not np.isfinite(pop).all():
        return None
    p = np.exp(log_p_pop(pop, np.full(n, n), np.zeros(n, dtype=int), pop_table))
    # 人気順（同率は馬番の小さい方を先に）。軸は1番人気
    order = np.lexsort((horse, pop))
    axis_i = order[0]
    if p[axis_i] < min_p:
        return None
    partners = [int(horse[i]) for i in order[1:1 + n_partners]]
    tickets = [(int(horse[axis_i]), b, c) for b, c in permutations(partners, 2)]
    return TrifectaPlan(int(horse[axis_i]), partners, float(p[axis_i]), tickets, unit_yen)
