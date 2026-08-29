"""鮮度ゲート（fail-closed）。

確定層の最新確定日が「前日」に届いていなければ、**推論を実行しない**。
古い履歴で計算した特徴量は学習時の分布から外れており、それで金額を賭けるのは
「推論が出ない」よりはるかに悪い（DB-04）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd

from ..clock import Clock, business_date, to_utc
from ..errors import StaleDataError
from .backend import Warehouse


@dataclass
class Freshness:
    latest_final_date: date | None
    required_date: date
    watermark: datetime | None
    is_fresh: bool

    def describe(self) -> str:
        return (f"確定層の最新 {self.latest_final_date}（要求 {self.required_date} 以降）"
                f" / watermark {self.watermark}")


# 鮮度判定で遡る日数。開催が飛ぶ期間があっても足りる長さにしておく。
# 長くしても走査するパーティションが増えるだけで、判定結果は変わらない。
FRESHNESS_LOOKBACK_DAYS = 90


def check(wh: Warehouse, clock: Clock, allow_same_day: bool = False,
          lookback_days: int = FRESHNESS_LOOKBACK_DAYS) -> Freshness:
    """前日ぶんが確定層に入っているか。"""
    today = business_date(clock.now())
    required = today if allow_same_day else today - timedelta(days=1)

    # 着順が入った行だけを見る。確定層には当月の未実施レースも入っているので、
    # 全行の MAX(race_date) を取ると「予定があるから最新」と誤判定する。
    #
    # `wh.con` は DuckDB 実装にしか無いので query() に寄せる。あわせて
    # race_date の範囲条件を必ず付ける。BigQuery 側は require_partition_filter
    # を有効にしてあり、条件の無い MAX(race_date) は実行そのものが拒否される
    # （本番の /health が 400 で落ちた）。直近だけ見れば鮮度は判定できる。
    since = required - timedelta(days=lookback_days)
    row = wh.query(
        "SELECT MAX(CASE WHEN finish_pos IS NOT NULL THEN race_date END) AS latest, "
        "MAX(merged_at) AS watermark FROM entry_result_final "
        "WHERE race_date >= ? AND race_date <= ?",
        [since, today], allow_full_scan=True)
    latest = row["latest"].iloc[0] if len(row) else None
    watermark = row["watermark"].iloc[0] if len(row) else None
    latest = None if pd.isna(latest) else latest
    watermark = None if pd.isna(watermark) else watermark
    if isinstance(latest, datetime):
        latest = latest.date()
    if isinstance(latest, pd.Timestamp):
        latest = latest.date()

    return Freshness(latest, required, watermark,
                     is_fresh=latest is not None and latest >= required)


def require_fresh(wh: Warehouse, clock: Clock, allow_same_day: bool = False) -> Freshness:
    f = check(wh, clock, allow_same_day)
    if not f.is_fresh:
        raise StaleDataError(
            "確定層が前日分を反映していないため推論を中止します（fail-closed）。"
            f" {f.describe()}")
    return f


def watermark(wh: Warehouse) -> datetime | None:
    """参照した DB の最終更新時刻。feature_snapshot に残す（設計書 §2.3）。"""
    # ここも同じ理由でパーティション条件が要る
    from datetime import date as _date

    since = _date.today() - timedelta(days=FRESHNESS_LOOKBACK_DAYS)
    rows = wh.query(
        "SELECT MAX(t) AS t FROM ("
        "SELECT MAX(merged_at) AS t FROM entry_result_final WHERE race_date >= ? "
        "UNION ALL "
        "SELECT MAX(captured_at) FROM entry_result_live WHERE race_date >= ?)",
        [since, since], allow_full_scan=True)
    value = rows["t"].iloc[0] if len(rows) else None
    return None if value is None or pd.isna(value) else to_utc(value)
