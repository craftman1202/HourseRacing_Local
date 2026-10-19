"""発注の安全装置。ここを通らない発注は無い。

1 日の上限は「賭け額の合計」ではなく「その日の損失」（2026-10-19、ユーザー指示 ¥3,000）。
  損失 = その日の始めの精算可能額 − 今の精算可能額（買った額は引かれ、払戻は戻る）＋ 処理中の買い目の額
  買い目を足してよいのは  損失 + この買い目の額 ≤ 上限  のとき（未確定の注文は全部外れると見なす）。
利益が出ていても上限は広げない（損失は 0 を下限にする。入金など残高の増減を利益と取り違えないため）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from .plan import OrderLine, OrderPlan

STATE_DIR = Path.home() / ".cache/nar-orders"
KILL_FILE = STATE_DIR / "STOP"          # 全ランナーを一時停止（touch するだけ）。個別は STOP.<名前>
LATE_AFTER_START_MIN = 15               # 発走からこれ以上過ぎた計画は、記録も通知もせず無視する


@dataclass(frozen=True)
class Limits:
    live: bool = False                       # false = ドライラン
    daily_loss_cap_yen: int = 0              # 1日の損失の上限
    race_cap_yen: int = 0                    # 1レースの上限（賭け額）
    line_cap_yen: int = 0                    # 1券種（1行）の上限（賭け額）
    bet_types: tuple[str, ...] = ()          # 発注を許す券種（空なら何も発注しない）
    open_min_before: float = 10.0            # 発走の何分前から
    close_min_before: float = 3.0            # 発走の何分前まで（締切前に余裕を持つ）
    max_consecutive_errors: int = 3

    @classmethod
    def load(cls, path: str | Path) -> "Limits":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if "daily_cap_yen" in raw:
            raise ValueError("daily_cap_yen（賭け額の上限）は廃止しました。daily_loss_cap_yen（損失の上限）にしてください")
        raw["bet_types"] = tuple(raw.get("bet_types", ()))
        return cls(**raw)


@dataclass
class Decision:
    allowed: list[OrderLine] = field(default_factory=list)
    skipped: list[tuple[OrderLine, str]] = field(default_factory=list)
    reason_all: str | None = None            # 全体を止める理由（窓外・停止ファイルなど）
    quiet: bool = False                      # True なら記録も通知もしない（一時停止中・発走後の古い計画）


def decide(plan: OrderPlan, limits: Limits, now: datetime, loss_so_far: int,
           already: set[tuple[str, str]], kill_files: tuple[Path, ...] = (KILL_FILE,)) -> Decision:
    """どの行を発注してよいか。止める理由は必ず文字で残す（通知に出す）。"""
    d = Decision()
    for kf in kill_files:
        if kf.exists():
            d.reason_all, d.quiet = f"停止ファイル {kf} があります", True
            return d
    start = plan.start()
    if now > start + timedelta(minutes=LATE_AFTER_START_MIN):
        d.reason_all, d.quiet = "発走後の古い計画", True
        return d
    if now > start - timedelta(minutes=limits.close_min_before):
        d.reason_all = f"締切が近い/過ぎている（発走 {limits.close_min_before:g} 分前を過ぎた）"
        return d
    if now < start - timedelta(minutes=limits.open_min_before):
        d.reason_all, d.quiet = "発注の時間窓の前", True
        return d
    loss = max(int(loss_so_far), 0)
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
        elif loss + race_total + ln.stake_yen > limits.daily_loss_cap_yen:
            d.skipped.append((ln, f"1日の損失上限 ¥{limits.daily_loss_cap_yen:,} 超"
                                  f"（今日の損失 ¥{loss:,} + この買い目 ¥{race_total + ln.stake_yen:,}）"))
        else:
            d.allowed.append(ln)
            race_total += ln.stake_yen
    return d
