"""発注台帳（sqlite）。二重発注の防止と、1日の発注額の集計。"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from .guard import STATE_DIR


class Ledger:
    def __init__(self, path: str | Path | None = None) -> None:
        import os

        self.path = Path(path or os.environ.get("NAR_ORDERS_LEDGER") or STATE_DIR / "ledger.sqlite")
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
        """その日の発注額。処理中（pending）も数える（別プロセスが今まさに買っているかもしれない）。"""
        status = "placed" if live else "dry_run"
        r = self.db.execute("SELECT COALESCE(SUM(stake_yen),0) FROM orders WHERE race_date=? "
                            "AND status IN (?, 'pending')", (race_date, status)).fetchone()
        return int(r[0])

    def claim(self, race_id: str, race_date: str, bet_type: str, tickets, stake_yen: int,
              at: datetime) -> bool:
        """この発注行を自分が処理すると宣言する。すでに誰かが宣言していたら False（発注しない）。

        排他ロックの代わり。主キー (race_id, bet_type) で原子的に決まるので、ランナーが
        別プロセスで並んで動いても（中央競馬用など）同じ買い目を 2 回買わない。
        落ちたプロセスが残した pending は自動では再試行しない（二重購入を避ける。人が履歴を見る）。
        """
        cur = self.db.execute("INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?,?,?)",
                              (race_id, race_date, bet_type, repr(tickets), stake_yen, "pending",
                               "発注処理中", at.isoformat()))
        self.db.commit()
        return cur.rowcount == 1

    def record(self, race_id: str, race_date: str, bet_type: str, tickets, stake_yen: int,
               status: str, detail: str, at: datetime) -> None:
        self.db.execute("INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?,?)",
                        (race_id, race_date, bet_type, repr(tickets), stake_yen, status, detail, at.isoformat()))
        self.db.commit()

    def rows(self, race_date: str):
        return self.db.execute("SELECT race_id, bet_type, stake_yen, status, detail FROM orders "
                               "WHERE race_date=? ORDER BY at", (race_date,)).fetchall()
