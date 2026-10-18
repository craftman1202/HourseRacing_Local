"""発注計画（クラウドが書き、PCのランナーが読む）。"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

log = logging.getLogger(__name__)
PREFIX = "orders"


@dataclass(frozen=True)
class OrderLine:
    bet_type: str                     # 単勝 / 複勝 / 枠連 / 馬連 / ワイド / 馬単 / 3連複 / 3連単 / 枠単
    tickets: tuple[tuple[int, ...], ...]   # 馬番（枠連・枠単は枠番）。馬単・3連単は着順どおり
    unit_yen: int
    structure: str = ""
    note: str = ""

    @property
    def stake_yen(self) -> int:
        return len(self.tickets) * self.unit_yen


@dataclass
class OrderPlan:
    race_id: str
    race_date: str                    # YYYY-MM-DD
    start_ts: str                     # ISO 8601 UTC
    track_name: str
    race_no: int
    release: str
    lines: list[OrderLine] = field(default_factory=list)

    @property
    def total_yen(self) -> int:
        return sum(ln.stake_yen for ln in self.lines)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "OrderPlan":
        d = json.loads(text)
        lines = [OrderLine(l["bet_type"], tuple(tuple(int(h) for h in t) for t in l["tickets"]),
                           int(l["unit_yen"]), l.get("structure", ""), l.get("note", ""))
                 for l in d.pop("lines")]
        return cls(lines=lines, **d)

    def start(self) -> datetime:
        return datetime.fromisoformat(self.start_ts)


def from_bet_plans(plans: list, *, race_id: str, race_date, start_ts, track_name: str,
                   race_no: int, release: str) -> OrderPlan:
    """推論の BetPlan（shared.BetPlan）から。stake=True の買い目だけが発注対象。"""
    lines = [OrderLine(pl.bet_type, tuple(tuple(int(h) for h in t) for t in pl.tickets),
                       int(pl.unit_yen), pl.structure, pl.note)
             for pl in plans if getattr(pl, "stake", True)]
    return OrderPlan(race_id, str(race_date), start_ts.isoformat(), track_name, int(race_no),
                     release, lines)


def blob_name(plan: OrderPlan) -> str:
    return f"{PREFIX}/{plan.race_date}/{plan.race_id}.json"


def publish_order_plan(plan: OrderPlan, bucket: str | None = None, project: str | None = None) -> bool:
    """GCS に置く。失敗しても推論は止めない（発注は付加機能）。bucket 未設定なら何もしない。"""
    bucket = bucket or os.environ.get("NAROPS_ORDER_BUCKET")
    if not bucket or not plan.lines:
        return False
    try:
        from google.cloud import storage

        client = storage.Client(project=project or os.environ.get("NAROPS_PROJECT"))
        client.bucket(bucket).blob(blob_name(plan)).upload_from_string(
            plan.to_json(), content_type="application/json")
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("発注計画の書き出しに失敗しました（%s）: %s", plan.race_id, str(exc)[:200])
        return False
