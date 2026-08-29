"""Too-Good-To-Be-True ガード（RF-01..08）。

競馬の市場効率性と控除率の水準を踏まえると、ここに挙げた数値は成功ではなく
バグの徴候。該当したら成果とみなさず、即座にリーク調査へ回す。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..errors import TooGoodToBeTrueError

BLOCKER = "Blocker"
CRITICAL = "Critical"


@dataclass
class Tripwire:
    id: str
    fired: bool
    severity: str
    observed: float | str
    message: str


def check(
    *,
    oos_top1: float | None = None,
    oos_nll: float | None = None,
    win_roi_uncorrected_monthly: pd.Series | None = None,
    uses_final_odds_for_decision: bool | None = None,
    cv_nll: float | None = None,
    feature_importance: pd.Series | None = None,
    shuffled_label_nll: float | None = None,
    max_drawdown: float | None = None,
) -> list[Tripwire]:
    out: list[Tripwire] = []

    def add(_id, fired, sev, obs, msg):
        out.append(Tripwire(_id, bool(fired), sev, obs, msg))

    if oos_top1 is not None:
        add("RF-01", oos_top1 > 0.60, BLOCKER, oos_top1,
            "OOS Top-1 が 60% 超。着順情報の混入がほぼ確実です。")
    if oos_nll is not None:
        add("RF-02", oos_nll < 1.20, BLOCKER, oos_nll,
            "OOS レース内 NLL が 1.20 未満。着順情報の混入がほぼ確実です。")
    if win_roi_uncorrected_monthly is not None and len(win_roi_uncorrected_monthly) >= 3:
        streak = _longest_streak(win_roi_uncorrected_monthly > 1.30)
        add("RF-03", streak >= 3, BLOCKER, streak,
            "補正前 単勝 ROI 130% 超が3か月以上継続。オッズ・払戻の時点混同を疑ってください。")
    if uses_final_odds_for_decision is not None:
        add("RF-04", uses_final_odds_for_decision, BLOCKER, uses_final_odds_for_decision,
            "賭け判断に確定オッズを使っています。締切前オッズとの取り違えです。")
    if cv_nll is not None and oos_nll is not None:
        gap = oos_nll - cv_nll
        add("RF-05", gap >= 0.10, CRITICAL, gap,
            "CV スコアが OOS より 0.10 以上良い。CV 分割のリークか過学習です。")
    if feature_importance is not None and len(feature_importance):
        share = float(feature_importance.max() / feature_importance.sum())
        add("RF-06", share > 0.40, CRITICAL, share,
            f"単一特徴量 {feature_importance.idxmax()!r} が importance の 40% 超。リーク源を疑ってください。")
    if shuffled_label_nll is not None:
        add("RF-07", shuffled_label_nll < 2.10, BLOCKER, shuffled_label_nll,
            "ラベルシャッフル下で NLL < 2.10。シャッフルしていない情報経路にリークがあります。")
    if max_drawdown is not None:
        add("RF-08", np.isclose(max_drawdown, 0.0, atol=1e-9), BLOCKER, max_drawdown,
            "最大ドローダウンがほぼ 0。損失の計上が抜けています。")
    return out


def _longest_streak(flags: pd.Series) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def enforce(tripwires: list[Tripwire]) -> None:
    """Blocker が発火したらパイプラインを止める。警告で済ませない。"""
    blockers = [t for t in tripwires if t.fired and t.severity == BLOCKER]
    if blockers:
        detail = "\n".join(f"  {t.id}: {t.message}（観測値 {t.observed}）" for t in blockers)
        raise TooGoodToBeTrueError(
            "良すぎる結果を検出しました。成果ではなくバグとして扱ってください。\n" + detail
        )


def to_frame(tripwires: list[Tripwire]) -> pd.DataFrame:
    return pd.DataFrame([t.__dict__ for t in tripwires])
