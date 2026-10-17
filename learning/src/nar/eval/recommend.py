"""全券種の推奨買い目（2026-10-13）。

検証（scripts/roi_search.py → roi_search_report.py → bet_selector.py、nested walk-forward）で
券種ごとに選んだ「買い方 × 確率の作り方 × 軸の確信度」を、運用の通知と学習側の検証が同じ
関数で組む（SK-02）。

確率の作り方: 本番の単勝確率（乖離回帰モデル）p_model と、人気だけの勝率 p_pop
（nar.eval.deviation.log_p_pop、人気×頭数の勝率表）を対数で混ぜてレース内 softmax:
    log π = (1−λ)·log p_model + λ·log p_pop
買い方: π の順位（同率は馬番の小さい方を先）で組む。軸 = π の1位。軸の π ≥ min_p のレースだけ買う。
枠連・枠単は馬の順位の組をその馬の枠番に置き換える（同じ枠どうしの組はゾロ目として残す）。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .deviation import log_p_pop

BET_JP = {"win": "単勝", "place": "複勝", "wide": "ワイド", "quinella": "馬連", "exacta": "馬単",
          "trio": "3連複", "trifecta": "3連単", "bracket_q": "枠連", "bracket_e": "枠単"}
ORDERED = {"win", "place", "exacta", "trifecta", "bracket_e"}
TOP = 9          # 組に使う順位の上限（9位まで）
MAX_MISSING = 2  # 人気の無い馬（取消・除外とみなす）をこの頭数まで除いて続ける


def structures() -> dict[str, tuple[str, list[tuple[int, ...]]]]:
    """買い方の名前 → (券種, 順位（0始まり）の組のリスト)。頭数が足りない組は使う側で落とす。"""
    S: dict[str, tuple[str, list]] = {}
    S["単勝 1位"] = ("win", [(0,)])
    S["単勝 1・2位"] = ("win", [(0,), (1,)])
    S["複勝 1位"] = ("place", [(0,)])
    S["複勝 1・2位"] = ("place", [(0,), (1,)])
    S["複勝 2位"] = ("place", [(1,)])
    for bt, jp in (("wide", "ワイド"), ("quinella", "馬連")):
        S[f"{jp} 1点"] = (bt, [(0, 1)])
        for n in (2, 4, 8):
            S[f"{jp} 軸1頭流し 相手{n}頭"] = (bt, [(0, j) for j in range(1, n + 1)])
        S[f"{jp} BOX3頭"] = (bt, list(itertools.combinations(range(3), 2)))
        S[f"{jp} BOX4頭"] = (bt, list(itertools.combinations(range(4), 2)))
    S["馬単 1点"] = ("exacta", [(0, 1)])
    S["馬単 裏表"] = ("exacta", [(0, 1), (1, 0)])
    for n in (2, 4, 8):
        S[f"馬単 1着固定流し 相手{n}頭"] = ("exacta", [(0, j) for j in range(1, n + 1)])
    for n in (2, 4, 7):
        S[f"3連複 軸2頭流し 相手{n}頭"] = ("trio", [(0, 1, k) for k in range(2, n + 2)])
    for n in (3, 5, 8):
        S[f"3連複 軸1頭流し 相手{n}頭"] = ("trio", list((0,) + c for c in itertools.combinations(range(1, n + 1), 2)))
    for n in (3, 4, 5):
        S[f"3連複 BOX{n}頭"] = ("trio", list(itertools.combinations(range(n), 3)))
    for n in (3, 5, 8):
        S[f"3連単 1着固定流し 相手{n}頭"] = ("trifecta", [(0,) + c for c in itertools.permutations(range(1, n + 1), 2)])
    for n in (2, 5):
        S[f"3連単 1・2着固定 3着{n}頭"] = ("trifecta", [(0, 1, k) for k in range(2, n + 2)])
    S["3連単 フォーメーション 1位/上位3/上位7"] = ("trifecta", [(0, j, k) for j in (1, 2) for k in range(1, 7) if k != j])
    S["3連単 BOX3頭"] = ("trifecta", list(itertools.permutations(range(3), 3)))
    # 2026-10-17 追加: 軸1頭＋相手3〜5頭の流し（既存の名前・中身は変えない）
    for n in (3, 5):
        S[f"ワイド 軸1頭流し 相手{n}頭"] = ("wide", [(0, j) for j in range(1, n + 1)])
    S["3連複 軸1頭流し 相手4頭"] = ("trio", list((0,) + c for c in itertools.combinations(range(1, 5), 2)))
    S["3連単 1着固定流し 相手4頭"] = ("trifecta", [(0,) + c for c in itertools.permutations(range(1, 5), 2)])
    for n in (3, 4, 5):
        S[f"3連単 軸1頭マルチ 相手{n}頭"] = ("trifecta", [
            t for c in itertools.permutations(range(1, n + 1), 2)
            for t in ((0, c[0], c[1]), (c[0], 0, c[1]), (c[0], c[1], 0))])
    S["枠連 1点"] = ("bracket_q", [(0, 1)])
    S["枠連 軸流し 相手3頭"] = ("bracket_q", [(0, j) for j in (1, 2, 3)])
    S["枠単 1点"] = ("bracket_e", [(0, 1)])
    S["枠単 1着固定流し 相手3頭"] = ("bracket_e", [(0, j) for j in (1, 2, 3)])
    return S


def tickets_for(structure: str, ranked_ids: np.ndarray) -> list[tuple[int, ...]]:
    """順位順に並んだ馬番（または枠番）から、その買い方の組（重複なし・券種の正規形）。"""
    bt, combos = structures()[structure]
    m = min(len(ranked_ids), TOP)
    out: list[tuple[int, ...]] = []
    seen = set()
    for c in combos:
        if max(c) >= m:
            continue
        vals = [ranked_ids[i] for i in c]
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
            continue
        t = tuple(int(v) for v in vals)
        if bt not in ORDERED:
            t = tuple(sorted(t))
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def blended_probability(p_model: np.ndarray, popularity: np.ndarray, pop_table: dict,
                        lam: float) -> np.ndarray:
    """1レース分の π。λ=0 で p_model、λ=1 で人気だけの勝率。"""
    p_model = np.asarray(p_model, dtype=float)
    n = len(p_model)
    lp_pop = log_p_pop(np.asarray(popularity, dtype=float), np.full(n, n), np.zeros(n, dtype=int), pop_table)
    lp = (1 - lam) * np.log(np.clip(p_model, 1e-12, 1)) + lam * lp_pop
    lp = lp - lp.max()
    p = np.exp(lp)
    return p / p.sum()


@dataclass(frozen=True)
class Rule:
    structure: str            # structures() のキー
    lam: float                # 人気側への寄せ（0〜1）
    min_p: float              # 軸の π の下限
    unit_yen: int = 100
    stake: bool = False       # True なら賭け金として記録する（日次上限に数える）。False は参考表示のみ
    note: str = ""            # 通知に添える検証値など
    standalone: bool = True   # False なら、他の理由で通知するレースにだけ添える（単独では通知しない）

    @property
    def bet(self) -> str:
        return structures()[self.structure][0]


@dataclass
class BetPlan:
    bet_type: str             # 券種（日本語）
    structure: str
    axis: int                 # 軸の馬番（枠の券種でも軸馬の馬番）
    partners: list[int]       # 相手（π の順、軸を除く上位）
    p_axis: float
    tickets: list[tuple[int, ...]] = field(default_factory=list)
    unit_yen: int = 100
    stake: bool = False
    note: str = ""
    standalone: bool = True
    # EV で決めた買い目の明細（ticket・ev・odds_hat・p）。通知に EV と予想配当を出すため
    detail: list = field(default_factory=list)

    @property
    def stake_yen(self) -> int:
        return len(self.tickets) * self.unit_yen


def recommend(horse_no: np.ndarray, waku: np.ndarray | None, p_model: np.ndarray,
              popularity: np.ndarray, pop_table: dict, rules: list[Rule]) -> list[BetPlan]:
    """1レース分。条件を満たす規則ごとに1プラン。

    人気（締切前オッズ）が無い馬は出走取消・除外とみなして除き、残りで確率を引き直す。
    無い馬が MAX_MISSING 頭を超える（オッズ未取得の疑い）・残りが2頭未満なら空。
    """
    horse = np.asarray(horse_no, dtype=int)
    pop = np.asarray(popularity, dtype=float)
    pm = np.asarray(p_model, dtype=float)
    ok = np.isfinite(pop) & np.isfinite(pm)
    if ok.sum() < 2 or (~ok).sum() > MAX_MISSING:
        return []
    horse, pop, pm = horse[ok], pop[ok], pm[ok] / pm[ok].sum()
    waku = None if waku is None else np.asarray(waku, dtype=float)[ok]
    p_model = pm
    n = len(horse)
    plans: list[BetPlan] = []
    cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for r in rules:
        if r.lam not in cache:
            p = blended_probability(p_model, pop, pop_table, r.lam)
            order = np.lexsort((horse, -p))
            cache[r.lam] = (p, order)
        p, order = cache[r.lam]
        if p[order[0]] < r.min_p:
            continue
        bt = r.bet
        if bt.startswith("bracket"):
            if waku is None:
                continue
            w = np.asarray(waku, dtype=float)
            ids = np.array([w[i] for i in order], dtype=object)
        else:
            ids = horse[order]
        t = tickets_for(r.structure, ids)
        if not t:
            continue
        plans.append(BetPlan(BET_JP[bt], r.structure, int(horse[order[0]]),
                             [int(h) for h in horse[order[1:TOP]]], float(p[order[0]]),
                             t, r.unit_yen, r.stake, r.note, r.standalone))
    return plans
