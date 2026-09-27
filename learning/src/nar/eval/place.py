"""複勝確率と「単勝・複勝の EV が高い方を1点」戦略。

学習側の検証（`scripts/place_model.py` / `scripts/place_win_strategies.py`）と
運用側の推論（`narops.inference`）が同じ関数を呼ぶ（SK-02）。

複勝専用モデルは3モデルとも「レース内の強さ」を出す。確率への変換は
  強さ → モデルごとの温度 → レース内 softmax → 重み付き対数合成 → 全体温度
  → Harville 式で「上位 k 着に入る確率」
で、レース内で Σ P(複勝) = k が厳密に成り立つ。k は出走頭数で決まる
（8頭以上 3 / 5〜7頭 2 / 4頭以下は複勝の発売なし = 0）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def place_slots(n_runners: np.ndarray | int) -> np.ndarray:
    """複勝の払戻対象になる着順の数。実払戻との一致 99.3%（1998〜2026 の複勝払戻で確認）。"""
    n = np.asarray(n_runners)
    return np.where(n >= 8, 3, np.where(n >= 5, 2, 0))


def _pad(values: np.ndarray, race_ids: np.ndarray):
    codes, uniq = pd.factorize(np.asarray(race_ids), sort=False)
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    pos = np.empty_like(codes)
    pos[order] = np.arange(len(codes)) - np.repeat(starts, sizes)
    grid = np.zeros((len(uniq), sizes.max()))
    grid[codes, pos] = values
    return grid, codes, pos


def harville_topk(p_win: np.ndarray, race_ids: np.ndarray, k: np.ndarray) -> np.ndarray:
    """単勝確率（レース内で総和 1）→ 上位 k 着（k ∈ {0,1,2,3}、レースごと）に入る確率。

    `metrics.harville_place_probability` と同じ量をベクトル化して求める（k が
    レースごとに違ってよい版）。
      P(i 2着) = Σ_j p_j · p_i / (1 - p_j)
      P(i 3着) = Σ_{j≠l} p_j · p_l / (1 - p_j) · p_i / (1 - p_j - p_l)
    3着項は B[j,l] = p_j p_l / ((1-p_j)(1-p_j-p_l)) の総和から i の行と列を引いて O(N²)。
    """
    p_win = np.asarray(p_win, dtype=float)
    if p_win.size == 0:
        return p_win.copy()
    p, codes, pos = _pad(p_win, race_ids)
    r = p / np.clip(1.0 - p, EPS, None)
    second = p * (r.sum(1, keepdims=True) - r)
    a = r[:, :, None] * p[:, None, :]
    b = a / np.clip(1.0 - p[:, :, None] - p[:, None, :], EPS, None)
    idx = np.arange(p.shape[1])
    b[:, idx, idx] = 0.0
    third = p * (b.sum((1, 2))[:, None] - b.sum(2) - b.sum(1))
    kk = np.zeros(p.shape[0])
    kk[codes] = np.asarray(k, dtype=float)
    place = (np.where(kk[:, None] >= 1, p, 0.0) + np.where(kk[:, None] >= 2, second, 0.0)
             + np.where(kk[:, None] >= 3, third, 0.0))
    return np.clip(place[codes, pos], 0.0, 1.0)


def race_log_softmax(scores: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(scores, dtype=float))
    z = s - s.groupby(race_ids).transform("max")
    return (z - np.log(np.exp(z).groupby(race_ids).transform("sum"))).to_numpy()


def place_probability(scores: dict[str, np.ndarray], temperatures: dict[str, float],
                      weights: dict[str, float], ensemble_temperature: float,
                      race_ids: np.ndarray, k: np.ndarray) -> np.ndarray:
    """複勝専用モデルの生スコア → P(複勝)。重み 0 のモデルは無くてよい。"""
    names = [m for m, w in weights.items() if w > 0]
    missing = [m for m in names if m not in scores]
    if missing:
        raise KeyError(f"複勝モデルの出力が足りません: {missing}")
    total = sum(weights[m] for m in names)
    blend = sum((weights[m] / total) * race_log_softmax(
        np.asarray(scores[m], dtype=float) / temperatures.get(m, 1.0), race_ids)
        for m in names)
    p_win_like = np.exp(race_log_softmax(blend / ensemble_temperature, race_ids))
    return harville_topk(p_win_like, race_ids, k)


def estimated_place_odds(pl_min: np.ndarray, pl_max: np.ndarray, alpha: float) -> np.ndarray:
    """複勝オッズの範囲 → 想定払戻倍率。

    実払戻は下限〜上限の範囲に 99.9% 入り、範囲内の平均位置は α≈0.257
    （2026-02〜05 の実払戻から推定、`scripts/place_win_strategies.py`）。
    """
    lo = np.asarray(pl_min, dtype=float)
    hi = np.asarray(pl_max, dtype=float)
    return lo + alpha * (hi - lo)


def kelly_stakes(p: np.ndarray, o: np.ndarray, eligible: np.ndarray, race_ids: np.ndarray,
                 budget: float, scale: float = 1.0, unit: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """券種ごとの予算に対するケリー配分。同一レースで Σf > 1 なら Σf = 1 に縮め、単位未満は切り捨て。"""
    p = np.asarray(p, dtype=float)
    o = np.asarray(o, dtype=float)
    ok = np.asarray(eligible, dtype=bool) & np.isfinite(o) & (o > 1.0) & np.isfinite(p)
    f = np.where(ok, (p * o - 1.0) / np.where(ok, o - 1.0, 1.0), 0.0) * scale
    f = np.clip(f, 0.0, None)
    tot = pd.Series(f).groupby(np.asarray(race_ids)).transform("sum").to_numpy()
    f = np.where(tot > 1.0, f / np.where(tot > 0, tot, 1.0), f)
    return np.floor(f * budget / unit) * unit, f


def max_ev_bets(race_ids: np.ndarray, p_win: np.ndarray, odds_win: np.ndarray,
                p_place: np.ndarray, place_odds: np.ndarray, *, min_ev: float = 1.0,
                min_prob: float = 0.6, budget_win: float = 10_000,
                budget_place: float = 10_000, kelly_scale: float = 1.0,
                unit: int = 100) -> pd.DataFrame:
    """馬ごとに単勝・複勝の EV の高い方を選び、EV≥min_ev かつその的中確率≥min_prob なら賭ける。

    金額は券種ごとの予算にケリー比率を掛けたもの。複勝の確率・オッズが無い馬は単勝だけで判定する。
    返り値の行順は入力と同じ。
    """
    rid = np.asarray(race_ids)
    pw, ow = np.asarray(p_win, float), np.asarray(odds_win, float)
    pp, op = np.asarray(p_place, float), np.asarray(place_odds, float)
    ev_w, ev_p = pw * ow, pp * op
    pick_win = ~(np.nan_to_num(ev_p, nan=-np.inf) > np.nan_to_num(ev_w, nan=-np.inf))
    ev = np.where(pick_win, ev_w, ev_p)
    prob = np.where(pick_win, pw, pp)
    eligible = np.isfinite(ev) & (ev >= min_ev) & (prob >= min_prob)
    s_w, f_w = kelly_stakes(pw, ow, eligible & pick_win, rid, budget_win, kelly_scale, unit)
    s_p, f_p = kelly_stakes(pp, op, eligible & ~pick_win, rid, budget_place, kelly_scale, unit)
    stake = np.where(pick_win, s_w, s_p)
    return pd.DataFrame({
        "ev_win": ev_w, "ev_place": ev_p,
        "bet_type": np.where(stake > 0, np.where(pick_win, "単勝", "複勝"), None),
        "stake_yen": stake.astype(int),
        "kelly": np.where(pick_win, f_w, f_p),
        "ev_chosen": ev,
    })
