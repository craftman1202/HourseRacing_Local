"""発注台帳（sqlite）。二重発注の防止と、1日の発注額の集計。"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from .guard import STATE_DIR


class Ledger:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or STATE_DIR / "ledger.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS orders (race_id TEXT, race_date TEXT, bet_type TEXT, "
            "tickets TEXT, stake_yen INT, status TEXT, detail TEXT, at TEXT, "
            "PRIMARY KEY (race_id, bet_type))")
        self.db.commit()

    def done(self, race_id: str) -> set[tuple[str, str]]:
        """この発注行は二度と出さない（placed / dry_run / error も含む。エラーの再試行は人が決める）。"""
        rows = self.db.execute("SELECT race_id, bet_type FROM orders WHERE race_id = ?", (race_id,))
        return set(rows.fetchall())

    def spent(self, race_date: str, live: bool) -> int:
        status = "placed" if live else "dry_run"
        r = self.db.execute("SELECT COALESCE(SUM(stake_yen),0) FROM orders WHERE race_date=? AND status=?",
                            (race_date, status)).fetchone()
        return int(r[0])

    def record(self, race_id: str, race_date: str, bet_type: str, tickets, stake_yen: int,
               status: str, detail: str, at: datetime) -> None:
        self.db.execute("INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?,?)",
                        (race_id, race_date, bet_type, repr(tickets), stake_yen, status, detail, at.isoformat()))
        self.db.commit()

    def rows(self, race_date: str):
        return self.db.execute("SELECT race_id, bet_type, stake_yen, status, detail FROM orders "
                               "WHERE race_date=? ORDER BY at", (race_date,)).fetchall()
