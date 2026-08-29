"""時刻の注入。

時刻依存ロジックは全て Clock 経由にする。テスト仕様 §1.3 のとおり freezegun は
ライブラリ内部の時刻取得を漏らすため、境界テストの主軸には使わない。

保存は UTC、営業日の境界判定は JST（DB-10 / WB-09）。この2つを混ぜると
00:00 JST 前後のレースが別の営業日に落ちる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Protocol

JST = timezone(timedelta(hours=9))
UTC = timezone.utc


class Clock(Protocol):
    def now(self) -> datetime:
        """常に tz-aware な UTC を返す。"""


@dataclass
class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


@dataclass
class FixedClock:
    """テスト用。tick() で任意に進める。"""

    current: datetime
    skew: timedelta = field(default_factory=timedelta)

    def now(self) -> datetime:
        return to_utc(self.current) + self.skew

    def tick(self, seconds: float) -> None:
        self.current = to_utc(self.current) + timedelta(seconds=seconds)


def to_utc(ts: datetime) -> datetime:
    """naive は JST とみなして UTC 化する。

    NAR のファイルに入っている時刻は JST の naive なので、素の datetime を
    UTC 扱いすると 9 時間ずれる。既定を JST 解釈にして事故を防ぐ。
    """
    return ts.replace(tzinfo=JST).astimezone(UTC) if ts.tzinfo is None else ts.astimezone(UTC)


def to_jst(ts: datetime) -> datetime:
    return to_utc(ts).astimezone(JST)


def business_date(ts: datetime) -> date:
    """営業日（JST の暦日）。"""
    return to_jst(ts).date()


def jst_datetime(*args, **kwargs) -> datetime:
    return datetime(*args, **kwargs, tzinfo=JST)


def minutes_until(target: datetime, clock: Clock) -> float:
    return (to_utc(target) - clock.now()).total_seconds() / 60.0
