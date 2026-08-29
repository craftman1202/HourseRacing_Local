"""ダウンロード台帳。

再ダウンロードを永久に不要にするための状態はすべてここに集約する。
should_fetch() が唯一の取得判断であり、ingest 側はこれを迂回しない。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS manifest (
  file_key       VARCHAR PRIMARY KEY,
  url            VARCHAR,
  params_json    VARCHAR,
  fetched_at     TIMESTAMP,
  http_status    INTEGER,
  content_length BIGINT,
  sha256         VARCHAR,
  raw_path       VARCHAR,
  inner_files    VARCHAR,
  schema_hash    VARCHAR,
  status         VARCHAR,
  is_final       BOOLEAN,
  unchanged_streak INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS manifest_history (
  file_key   VARCHAR,
  fetched_at TIMESTAMP,
  sha256     VARCHAR,
  raw_path   VARCHAR
);
"""


@dataclass
class Record:
    file_key: str
    url: str = ""
    params_json: str = "{}"
    fetched_at: Any = None
    http_status: int = 0
    content_length: int = 0
    sha256: str = ""
    raw_path: str = ""
    inner_files: str = ""
    schema_hash: str = ""
    status: str = "ok"
    is_final: bool = False
    unchanged_streak: int = 0


class Manifest:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(self.db_path)
        self.con.execute(SCHEMA)

    def close(self) -> None:
        self.con.close()

    def get(self, file_key: str) -> Record | None:
        row = self.con.execute(
            "SELECT file_key,url,params_json,fetched_at,http_status,content_length,"
            "sha256,raw_path,inner_files,schema_hash,status,is_final,unchanged_streak "
            "FROM manifest WHERE file_key = ?",
            [file_key],
        ).fetchone()
        return Record(*row) if row else None

    def upsert(self, rec: Record) -> None:
        prev = self.get(rec.file_key)
        if prev is not None and prev.sha256:
            self.con.execute(
                "INSERT INTO manifest_history VALUES (?,?,?,?)",
                [prev.file_key, prev.fetched_at, prev.sha256, prev.raw_path],
            )
            # ハッシュが変われば「確定」判断のカウンタをリセットする（IG-13）
            rec.unchanged_streak = (
                prev.unchanged_streak + 1 if prev.sha256 == rec.sha256 else 0
            )
        self.con.execute("DELETE FROM manifest WHERE file_key = ?", [rec.file_key])
        self.con.execute(
            "INSERT INTO manifest VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                rec.file_key, rec.url, rec.params_json, rec.fetched_at, rec.http_status,
                rec.content_length, rec.sha256, rec.raw_path, rec.inner_files,
                rec.schema_hash, rec.status, rec.is_final, rec.unchanged_streak,
            ],
        )

    def all(self) -> list[Record]:
        rows = self.con.execute(
            "SELECT file_key,url,params_json,fetched_at,http_status,content_length,"
            "sha256,raw_path,inner_files,schema_hash,status,is_final,unchanged_streak "
            "FROM manifest ORDER BY file_key"
        ).fetchall()
        return [Record(*r) for r in rows]

    def dataset_version(self) -> str:
        """raw が1バイトでも変われば変わるバージョン識別子（RP-03）。"""
        import hashlib

        h = hashlib.sha256()
        for rec in self.all():
            h.update(rec.file_key.encode())
            h.update(rec.sha256.encode())
        return h.hexdigest()


def month_end(ym: str) -> date:
    y, m = (int(x) for x in ym.split("-"))
    return date(y + m // 12, m % 12 + 1, 1) - timedelta(days=1)


def is_finalizable(ym: str, today: date, finalize_after_days: int) -> bool:
    return (today - month_end(ym)).days >= finalize_after_days


def should_fetch(
    manifest: Manifest,
    file_key: str,
    ym: str,
    today: date,
    finalize_after_days: int = 45,
    daily_refresh_months: int = 2,
) -> bool:
    """取得すべきか。

    確定済み（is_final）は無条件でスキップ。未確定は「当月＋直近 N-1 か月」だけを
    見に行く。344 か月を毎回叩かないのが再ダウンロード回避の実体（IG-16）。
    """
    rec = manifest.get(file_key)
    if rec is None or rec.status != "ok":
        return True
    if rec.is_final:
        return False
    return ym >= _shift_ym(today, -(daily_refresh_months - 1))


def _shift_ym(d: date, months: int) -> str:
    total = d.year * 12 + (d.month - 1) + months
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def finalize_pass(manifest: Manifest, today: date, finalize_after_days: int = 45) -> int:
    """月末+N日を経過し、かつ直近取得でハッシュ不変だったものを確定させる（IG-12）。"""
    n = 0
    for rec in manifest.all():
        if rec.is_final or rec.status != "ok":
            continue
        ym = rec.file_key.rsplit("/", 1)[-1][:7]
        if not _is_ym(ym):
            continue
        if is_finalizable(ym, today, finalize_after_days) and rec.unchanged_streak >= 1:
            rec.is_final = True
            manifest.con.execute(
                "UPDATE manifest SET is_final = TRUE WHERE file_key = ?", [rec.file_key]
            )
            n += 1
    return n


def _is_ym(s: str) -> bool:
    return len(s) == 7 and s[4] == "-" and s[:4].isdigit() and s[5:].isdigit()
