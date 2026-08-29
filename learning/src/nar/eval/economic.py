"""経済指標。

自己インパクト補正は任意ではなく必須。地方の小規模プールでは、補正しただけで
期待値が消えるケースが頻出する。補正前後の両方を報告し、差分を明示する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NOMINAL_TAKEOUT = {
    "単勝": 0.20, "複勝": 0.20, "枠連": 0.25, "馬連": 0.25,
    "馬単": 0.25, "ワイド": 0.25, "3連複": 0.25, "3連単": 0.25,
}


def implied_takeout(odds: np.ndarray) -> float:
    """オッズ逆数和 Σ1/o = 1/(1-τ) から τ を逆算する（EC-01..03）。"""
    s = float(np.sum(1.0 / np.asarray(odds, dtype=float)))
    return 1.0 - 1.0 / s


def inverse_odds_sum(odds: np.ndarray) -> float:
    return float(np.sum(1.0 / np.asarray(odds, dtype=float)))


def effective_odds(o: float, b: float, pool: float, takeout: float) -> float:
    """自己インパクト補正後の実効オッズ。

        S = P(1-τ)/o   （当該組み合わせへの既存投票額）
        o_eff = (P + b)(1-τ) / (S + b)

    b=0 で o に一致し、b の増加に対して単調減少する（EC-05/06）。
    """
    if o <= 0:
        raise ValueError("オッズは正である必要があります")
    if b < 0:
        raise ValueError("投票額は非負である必要があります")
    s = pool * (1.0 - takeout) / o
    return (pool + b) * (1.0 - takeout) / (s + b)


def kelly_fraction(p: float | np.ndarray, o: float | np.ndarray, cap: float = 0.25) -> np.ndarray:
    """f* = (po - 1) / (o - 1) を cap 倍したもの。負（期待値マイナス）は 0 に潰す。"""
    p = np.asarray(p, dtype=float)
    o = np.asarray(o, dtype=float)
    full = (p * o - 1.0) / (o - 1.0)
    return np.clip(full, 0.0, None) * cap


@dataclass
class BetResult:
    n_bets: int
    stake: float
    ret: float
    roi: float
    hit_rate: float
    max_drawdown: float


def simulate_win_bets(
    df: pd.DataFrame, p_col: str = "p", odds_col: str = "odds_win",
    win_col: str = "is_win", date_col: str = "race_date",
    ev_threshold: float = 1.0, stake: float = 100.0,
    pool: float | None = None, takeout: float = 0.20,
) -> tuple[BetResult, pd.DataFrame]:
    """単勝の期待値閾値ベット。

    pool を渡すと自己インパクト補正後のオッズで払戻を計算する。補正後 ROI が
    補正前を上回ることは構造上ありえない（EC-07）。
    """
    work = df.copy()
    work["ev"] = work[p_col] * work[odds_col]
    bets = work[work["ev"] > ev_threshold].copy()
    if bets.empty:
        return BetResult(0, 0.0, 0.0, np.nan, np.nan, 0.0), bets

    if pool is not None:
        bets["odds_used"] = [
            effective_odds(o, stake, pool, takeout) for o in bets[odds_col]
        ]
    else:
        bets["odds_used"] = bets[odds_col]

    bets["stake"] = stake
    bets["ret"] = np.where(bets[win_col] == 1, stake * bets["odds_used"], 0.0)
    bets = bets.sort_values(date_col)
    bets["cum_pnl"] = (bets["ret"] - bets["stake"]).cumsum()

    total_stake = float(bets["stake"].sum())
    total_ret = float(bets["ret"].sum())
    return (
        BetResult(
            n_bets=len(bets),
            stake=total_stake,
            ret=total_ret,
            roi=total_ret / total_stake,
            hit_rate=float((bets[win_col] == 1).mean()),
            max_drawdown=max_drawdown(bets["cum_pnl"]),
        ),
        bets,
    )


def max_drawdown(equity: pd.Series) -> float:
    """資金曲線の最大ドローダウン。非負で返す（EC-12）。"""
    eq = equity.to_numpy(dtype=float)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    return float(np.max(peak - eq))


def flat_bet_all_roi(df: pd.DataFrame, odds_col="odds_win", win_col="is_win") -> float:
    """全馬に均等投票したときの ROI。理論上 (1-τ) になる（EC-08）。

    経済評価パイプライン全体のサニティチェック。80% から大きく外れるなら、
    払戻の突合か控除率の扱いが間違っている。
    """
    ret = float((df[win_col] * df[odds_col]).sum())
    return ret / len(df)


def measured_takeout_by_track(odds: pd.DataFrame, bet_type: str = "単勝") -> pd.DataFrame:
    """場別の控除率実測。主催者ごとに設定が違うので、期待値計算には実測値を使う。"""
    rows = []
    for (baba, rid), g in odds.groupby(["baba_code", "race_id"]):
        rows.append({"baba_code": baba, "race_id": rid,
                     "inv_sum": inverse_odds_sum(g["odds_win"].to_numpy())})
    df = pd.DataFrame(rows)
    out = df.groupby("baba_code")["inv_sum"].agg(["mean", "std", "count"]).reset_index()
    out["takeout_measured"] = 1.0 - 1.0 / out["mean"]
    out["takeout_nominal"] = NOMINAL_TAKEOUT[bet_type]
    out["diff_pt"] = (out["takeout_measured"] - out["takeout_nominal"]) * 100
    return out
