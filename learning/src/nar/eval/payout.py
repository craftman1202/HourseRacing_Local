"""払戻（オッズ）の予測と、EV で買い目を決める（2026-10-17）。

学習側の検証（scripts/payout_ev.py）と運用側の通知が同じ関数を使う（SK-02）。

  組の確率   Harville（勝率ベクトル → 馬連・馬単・ワイド・3連複・3連単、複勝は上位 k 着）
  払戻予測   券種ごとの LightGBM（log 払戻）。特徴量は組の各馬の単勝人気・走った頭数・
             人気だけの勝率表から出した組の確率 log P_pop・競馬場（payout_features）。
             E[払戻] = exp(予測) × smear
  EV         P_ens(組) × 予測払戻。P_ens はアンサンブルの勝率から Harville で出す
  買い目     EvRule: EV ≥ t の組を「全部（all）」か「レースで EV 最大の1点（top1）」
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .place import harville_topk, place_slots
from .recommend import BET_JP, BetPlan

K = {"win": 1, "place": 1, "quinella": 2, "exacta": 2, "wide": 2, "trio": 3, "trifecta": 3}
ORDERED = {"win", "place", "exacta", "trifecta"}
TOP = 9


def harville_all(bt: str, P: np.ndarray) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """(R, N) の勝率（列は確率の降順、パディングは 0）→ 上位 TOP 位置の全組の確率 (R, C) と組の位置。

    wide の 3 頭目は全頭で足す。place は呼び出し側で harville_topk を使う。
    """
    Rn, N = P.shape
    m = min(TOP, N)
    eps = 1e-12
    if bt == "win":
        return P[:, :m], [(i,) for i in range(m)]
    if bt in ("exacta", "quinella"):
        perms = list(itertools.permutations(range(m), 2))
        ex = {q: P[:, q[0]] * P[:, q[1]] / np.maximum(1 - P[:, q[0]], eps) for q in perms}
        if bt == "exacta":
            return np.stack([ex[q] for q in perms], 1), perms
        cs = list(itertools.combinations(range(m), 2))
        return np.stack([ex[(a, b)] + ex[(b, a)] for a, b in cs], 1), cs

    def tri(a, b, c):
        return (P[:, a] * P[:, b] / np.maximum(1 - P[:, a], eps)
                * P[:, c] / np.maximum(1 - P[:, a] - P[:, b], eps))
    if bt == "trifecta":
        perms = list(itertools.permutations(range(m), 3))
        return np.stack([tri(*q) for q in perms], 1), perms
    if bt == "trio":
        cs = list(itertools.combinations(range(m), 3))
        return np.stack([sum(tri(*q) for q in itertools.permutations(c)) for c in cs], 1), cs
    if bt == "wide":
        cs = list(itertools.combinations(range(m), 2))
        out = []
        for a, b in cs:
            acc = np.zeros(Rn)
            for k in range(N):
                if k in (a, b):
                    continue
                acc += sum(tri(*q) for q in itertools.permutations((a, b, k)))
            out.append(acc)
        return np.stack(out, 1), cs
    raise ValueError(bt)


def payout_features(bt: str, ranks: np.ndarray, n: np.ndarray, logp_pop: np.ndarray,
                    track: np.ndarray) -> pd.DataFrame:
    """ranks: (rows, K) 組の各馬の単勝人気（順序なし券種は昇順に並べ替え済み）。"""
    f = {f"r{i + 1}": ranks[:, i] for i in range(ranks.shape[1])}
    f.update({"n": n, "logp_pop": logp_pop, "track": track, "rsum": ranks.sum(1)})
    if bt in ("place", "wide"):
        f["slots"] = np.where(n >= 8, 3, 2)
    return pd.DataFrame(f)


@dataclass
class PayoutModel:
    bet: str
    booster: object          # lightgbm.Booster
    smear: float

    def predict_odds(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(np.asarray(self.booster.predict(X))) * self.smear


def race_tickets(bt: str, horse_no: np.ndarray, p_ens: np.ndarray, p_pop: np.ndarray,
                 popularity: np.ndarray, track: int, model: PayoutModel) -> pd.DataFrame:
    """1 レース分の全組（確率上位 TOP 頭から）の P_ens・予測払戻・EV。

    horse_no / p_ens / p_pop / popularity は同じ並び（走る馬全頭）。確率は総和 1。
    戻り値の列: tickets（馬番のタプル。順序なし券種は昇順）, positions, p_ens, p_pop, odds_hat, ev
    """
    order = np.lexsort((np.asarray(horse_no), -np.asarray(p_ens)))
    hn = np.asarray(horse_no, int)[order]
    pe = np.asarray(p_ens, float)[order]
    pp = np.asarray(p_pop, float)[order]
    rk = np.clip(np.nan_to_num(np.asarray(popularity, float)[order], nan=len(hn)), 1, 16).astype(int)
    n = len(hn)
    if bt == "place":
        k = int(place_slots(n))
        if k == 0:
            return pd.DataFrame()
        m = min(TOP, n)
        ce = harville_topk(pe, np.zeros(n), np.full(n, k))[:m][None, :]
        cp = harville_topk(pp, np.zeros(n), np.full(n, k))[:m][None, :]
        combos = [(i,) for i in range(m)]
    else:
        if n < K[bt]:
            return pd.DataFrame()
        ce, combos = harville_all(bt, pe[None, :])
        cp, _ = harville_all(bt, pp[None, :])
    idx = np.array(combos)
    ranks = rk[idx]
    hns = hn[idx]
    if bt not in ORDERED:
        o = np.argsort(hns, axis=1)
        hns = np.take_along_axis(hns, o, 1)
        ranks = np.sort(ranks, axis=1)
    X = payout_features(bt, ranks, np.full(len(idx), n), np.log(np.maximum(cp[0], 1e-9)),
                        np.full(len(idx), int(track)))
    odds = model.predict_odds(X)
    return pd.DataFrame({"tickets": [tuple(int(h) for h in r) for r in hns],
                         "p_ens": ce[0], "p_pop": cp[0], "odds_hat": odds, "ev": ce[0] * odds})


@dataclass(frozen=True)
class EvRule:
    bet: str                  # win / place / quinella / exacta / wide / trio / trifecta
    t: float                  # EV の下限
    mode: str = "top1"        # "top1"（レースで EV 最大の1点）/ "all"（条件を満たす全部）
    unit_yen: int = 100
    stake: bool = True
    note: str = ""
    standalone: bool = True


def ev_plans(horse_no, p_ens, p_pop, popularity, track: int, models: dict[str, PayoutModel],
             rules: list[EvRule]) -> list[BetPlan]:
    """EV ルールごとの買い目。人気が欠けた馬は呼び出し側で除いておく（recommend と同じ扱い）。

    BetPlan の axis は「買い目でいちばん確率の高い馬」（1 点なら P_ens 最大の馬、複数点なら
    最も多くの組に入る馬）、partners は残り。tickets は EV の高い順。
    """
    plans: list[BetPlan] = []
    pe_by_horse = dict(zip(np.asarray(horse_no, int), np.asarray(p_ens, float)))
    for r in rules:
        m = models.get(r.bet)
        if m is None:
            continue
        tk = race_tickets(r.bet, horse_no, p_ens, p_pop, popularity, track, m)
        if tk.empty:
            continue
        sel = tk[tk["ev"] >= r.t].sort_values("ev", ascending=False)
        if sel.empty:
            continue
        if r.mode == "top1":
            sel = sel.head(1)
        tickets = list(sel["tickets"])
        count: dict[int, int] = {}
        for t in tickets:
            for h in t:
                count[h] = count.get(h, 0) + 1
        axis = max(count, key=lambda h: (count[h], pe_by_horse.get(h, 0.0)))
        partners = [h for h in sorted(count, key=lambda h: -pe_by_horse.get(h, 0.0)) if h != axis]
        detail = [{"ticket": t, "ev": float(e), "odds_hat": float(o), "p": float(p)}
                  for t, e, o, p in zip(sel["tickets"], sel["ev"], sel["odds_hat"], sel["p_ens"])]
        plan = BetPlan(BET_JP[r.bet], f"{BET_JP[r.bet]} EV≥{r.t:g}" + ("（最大1点）" if r.mode == "top1" else "（全部）"),
                       int(axis), [int(h) for h in partners], float(pe_by_horse.get(axis, 0.0)),
                       tickets, r.unit_yen, r.stake, r.note, r.standalone, detail)
        plans.append(plan)
    return plans
