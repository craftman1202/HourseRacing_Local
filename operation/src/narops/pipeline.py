"""推論結果の永続化と、失敗時の明示的な分岐。

すべての書き込みは `(race_id, model_release)` を主キーとする MERGE 相当で、
同一タスクの再実行が重複を生まない（IN-08 / DR-06）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd

from .clock import Clock
from .db.backend import Warehouse
from .db.types import canonicalize_frame

Status = Literal["ok", "insufficient_data", "expired", "not_computed"]


@dataclass
class InferenceOutcome:
    """1レース分の結果。失敗も「結果」として明示的に持つ。

    失敗を例外だけで表すと、Web 側が「まだ計算していない」のか
    「計算して候補ゼロだった」のかを区別できない。
    """

    race_id: str
    status: Status
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    bet_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    reason: str = ""
    retryable: bool = False

    @property
    def web_label(self) -> str:
        return {"ok": "算出済み", "insufficient_data": "データ不足",
                "expired": "締切超過", "not_computed": "未算出"}[self.status]

    @classmethod
    def failed(cls, race_id: str, reason: str, retryable: bool = False,
               status: Status = "not_computed") -> "InferenceOutcome":
        return cls(race_id, status, pd.DataFrame(), pd.DataFrame(), reason, retryable)


PREDICTION_COLUMNS = ["race_id", "horse_no", "race_date", "model_release",
                      "track_used", "p_win", "p_market", "ev", "ev_adjusted",
                      "computed_at", "is_shadow"]
BET_COLUMNS = ["race_id", "horse_no", "race_date", "model_release", "bet_type",
               "stake_yen", "ev", "ev_adjusted", "kelly", "computed_at"]


def write_prediction(wh: Warehouse, race_id: str, frame: pd.DataFrame,
                     model_release: str, track_used: str, race_date: date,
                     clock: Clock, is_shadow: bool = False) -> int:
    """予測の書き込み。同一 (race_id, model_release) は置き換える。

    追記にすると再実行のたびに行が増え、どれが最終版か分からなくなる。
    """
    payload = pd.DataFrame({
        "race_id": race_id,
        "horse_no": frame["horse_no"].astype(int),
        "race_date": race_date,
        "model_release": model_release,
        "track_used": track_used,
        "p_win": frame["p_win"].astype(float),
        "p_market": frame.get("p_market", pd.Series([None] * len(frame))),
        "ev": frame.get("ev", pd.Series([None] * len(frame))),
        "ev_adjusted": frame.get("ev_adjusted", pd.Series([None] * len(frame))),
        "computed_at": clock.now(),
        "is_shadow": is_shadow,
    })[PREDICTION_COLUMNS]

    wh.execute("DELETE FROM prediction WHERE race_date = ? AND race_id = ? "
               "AND model_release = ? AND is_shadow = ?",
               [race_date, race_id, model_release, is_shadow])
    wh.insert_frame("prediction", canonicalize_frame(payload))
    return len(payload)


def write_bet_candidates(wh: Warehouse, race_id: str, frame: pd.DataFrame,
                         model_release: str, race_date: date, clock: Clock,
                         bet_type: str = "単勝") -> int:
    """ベット候補。シャドー版は呼んではいけない（RL-02 は release 側で強制）。"""
    bets = frame[frame["stake_yen"] > 0]
    wh.execute("DELETE FROM bet_candidate WHERE race_date = ? AND race_id = ? "
               "AND model_release = ?", [race_date, race_id, model_release])
    if bets.empty:
        return 0
    payload = pd.DataFrame({
        "race_id": race_id, "horse_no": bets["horse_no"].astype(int),
        "race_date": race_date, "model_release": model_release, "bet_type": bet_type,
        "stake_yen": bets["stake_yen"].astype(int),
        "ev": bets["ev"].astype(float),
        "ev_adjusted": bets["ev_adjusted"].astype(float),
        "kelly": bets.get("kelly", pd.Series([0.0] * len(bets))).astype(float),
        "computed_at": clock.now(),
    })[BET_COLUMNS]
    wh.insert_frame("bet_candidate", canonicalize_frame(payload))
    return len(payload)


def day_budget_remaining(wh: Warehouse, business_day: date, max_per_day: int) -> int:
    """本日の残枠。上限は環境変数で管理し、コードに埋め込まない（設計書 §3.3）。"""
    row = wh.query(
        "SELECT COALESCE(SUM(stake_yen), 0) AS used FROM bet_candidate "
        "WHERE race_date = ?", [business_day], allow_full_scan=True)
    used = int(row["used"].iloc[0]) if len(row) else 0
    return max(0, max_per_day - used)
