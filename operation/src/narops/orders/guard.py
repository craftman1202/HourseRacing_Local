"""発注の安全装置。ここを通らない発注は無い。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from .plan import OrderLine, OrderPlan

STATE_DIR = Path.home() / ".cache/nar-orders"
KILL_FILE = STATE_DIR / "STOP"          # このファイルがあれば全停止（touch するだけでよい）


@dataclass(frozen=True)
class Limits:
    live: bool = False                       # false = ドライラン
    daily_cap_yen: int = 0                   # 1日の発注合計の上限
    race_cap_yen: int = 0                    # 1レースの上限
    line_cap_yen: int = 0                    # 1券種（1行）の上限
    bet_types: tuple[str, ...] = ()          # 発注を許す券種（空なら何も発注しない）
    open_min_before: float = 10.0            # 発走の何分前から
    close_min_before: float = 3.0            # 発走の何分前まで（締切前に余裕を持つ）
    max_consecutive_errors: int = 3

    @classmethod
    def load(cls, path: str | Path) -> "Limits":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw["bet_types"] = tuple(raw.get("bet_types", ()))
        return cls(**raw)


@dataclass
class Decision:
    allowed: list[OrderLine] = field(default_factory=list)
    skipped: list[tuple[OrderLine, str]] = field(default_factory=list)
    reason_all: str | None = None            # 全体を止める理由（窓外・停止ファイルなど）


def decide(plan: OrderPlan, limits: Limits, now: datetime, spent_today: int,
           already: set[tuple[str, str]], kill_file: Path = KILL_FILE) -> Decision:
    """どの行を発注してよいか。止める理由は必ず文字で残す（通知に出す）。"""
    d = Decision()
    if kill_file.exists():
        d.reason_all = f"停止ファイル {kill_file} があります"
        return d
    start = plan.start()
    if now > start - timedelta(minutes=limits.close_min_before):
        d.reason_all = f"締切が近い/過ぎている（発走 {limits.close_min_before:g} 分前を過ぎた）"
        return d
    if now < start - timedelta(minutes=limits.open_min_before):
        d.reason_all = "発注の時間窓の前"
        return d
    race_total = 0
    for ln in plan.lines:
        key = (plan.race_id, ln.bet_type)
        if key in already:
            d.skipped.append((ln, "発注済み"))
        elif ln.bet_type not in limits.bet_types:
            d.skipped.append((ln, f"{ln.bet_type} は発注対象に入れていません"))
        elif ln.unit_yen % 100 or ln.unit_yen <= 0 or not ln.tickets:
            d.skipped.append((ln, "金額・点数が不正"))
        elif ln.stake_yen > limits.line_cap_yen:
            d.skipped.append((ln, f"1券種の上限 ¥{limits.line_cap_yen:,} 超（¥{ln.stake_yen:,}）"))
        elif race_total + ln.stake_yen > limits.race_cap_yen:
            d.skipped.append((ln, f"1レースの上限 ¥{limits.race_cap_yen:,} 超"))
        elif spent_today + race_total + ln.stake_yen > limits.daily_cap_yen:
            d.skipped.append((ln, f"1日の上限 ¥{limits.daily_cap_yen:,} 超"))
        else:
            d.allowed.append(ln)
            race_total += ln.stake_yen
    return d
