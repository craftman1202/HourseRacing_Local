"""月次ファイルの取得。

原本 ZIP をバイト列のまま不変に保存し、そこから下流をすべて再生成可能にする。
パース仕様やスキーマ定義を変更しても、NAR への再アクセスは発生しない（設計書 §1）。

実測したエンドポイント形式（2026-08 時点）:
    GET .../DataDownload/RaceDataDownload?type=monthly&k_year=2026&k_month=7
    → application/zip、中身は {YYYYMM}_racelist.csv / _payback.csv / _horselist.csv
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime

from ..io.manifest import Manifest, Record, is_finalizable, should_fetch
from ..io.store import Store, sha256_bytes
from .client import NarClient
from .unzip import extract, inner_names

log = logging.getLogger(__name__)

RACE_URL = "https://www.keiba.go.jp/KeibaWeb/DataDownload/RaceDataDownload"
ODDS_URL = "https://www.keiba.go.jp/KeibaWeb/DataDownload/OddsDataDownload"


@dataclass
class FetchOutcome:
    file_key: str
    status: str          # downloaded / unchanged / skipped / error
    sha256: str = ""
    size: int = 0
    inner: tuple[str, ...] = ()
    error: str = ""


def month_range(start_ym: str, end_ym: str) -> list[str]:
    """'1998-01' 〜 '2026-08' の全月。"""
    sy, sm = (int(x) for x in start_ym.split("-"))
    ey, em = (int(x) for x in end_ym.split("-"))
    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def race_file_key(ym: str) -> str:
    return f"monthly/race/{ym}"


def odds_file_key(ym: str, part: int) -> str:
    return f"monthly/odds/{ym}/{part:02d}"


def fetch_month(
    client: NarClient, store: Store, manifest: Manifest, ym: str,
    today: date, finalize_after_days: int = 45, daily_refresh_months: int = 2,
    kind: str = "race", part: int | None = None,
) -> FetchOutcome:
    """1か月分の取得。

    確定済み（is_final）はネットワークに触らない。ハッシュが前回と同じなら
    ディスク書き込みごとスキップする（IG-02/03）。
    """
    key = race_file_key(ym) if kind == "race" else odds_file_key(ym, part or 1)
    if not should_fetch(manifest, key, ym, today, finalize_after_days,
                        daily_refresh_months):
        return FetchOutcome(key, "skipped")

    y, m = (int(x) for x in ym.split("-"))
    url = RACE_URL if kind == "race" else ODDS_URL
    # 実測: オッズも `type=monthly` の1リクエストで3分割ぶんすべてが1つの ZIP に入る。
    # 設計書は10日区切り3分割で18ファイルを想定していたが、実際は月6リクエストで済む。
    params: dict = {"type": "monthly", "k_year": y, "k_month": m}

    try:
        res = client.fetch(url, params)
    except Exception as exc:  # noqa: BLE001
        manifest.upsert(Record(file_key=key, url=url, params_json=json.dumps(params),
                               fetched_at=datetime.now(), status="parse_error"))
        return FetchOutcome(key, "error", error=str(exc)[:200])

    digest = sha256_bytes(res.content)
    prev = manifest.get(key)
    if prev is not None and prev.sha256 == digest:
        # 内容不変。ディスクは触らず、確定判定のためのカウンタだけ進める
        manifest.upsert(Record(
            file_key=key, url=url, params_json=json.dumps(params),
            fetched_at=datetime.now(), http_status=res.status_code,
            content_length=len(res.content), sha256=digest, raw_path=prev.raw_path,
            inner_files=prev.inner_files, schema_hash=prev.schema_hash, status="ok",
            is_final=is_finalizable(ym, today, finalize_after_days)))
        return FetchOutcome(key, "unchanged", digest, len(res.content))

    sub = "race" if kind == "race" else "odds"
    fname = (f"{ym.replace('-', '')}_{sub}.zip" if part is None
             else f"{ym.replace('-', '')}_{part:02d}_{sub}.zip")
    path = store.write_atomic(res.content, "raw", "monthly", sub, f"ym={ym}", fname)

    try:
        names = tuple(inner_names(res.content))
    except Exception as exc:  # noqa: BLE001
        manifest.upsert(Record(file_key=key, url=url, params_json=json.dumps(params),
                               fetched_at=datetime.now(), http_status=res.status_code,
                               content_length=len(res.content), sha256=digest,
                               raw_path=path, status="parse_error"))
        return FetchOutcome(key, "error", digest, len(res.content),
                            error=f"ZIP を開けません: {exc}")

    manifest.upsert(Record(
        file_key=key, url=url, params_json=json.dumps(params),
        fetched_at=datetime.now(), http_status=res.status_code,
        content_length=len(res.content), sha256=digest, raw_path=path,
        inner_files=json.dumps(list(names), ensure_ascii=False), status="ok",
        is_final=False))
    return FetchOutcome(key, "downloaded", digest, len(res.content), names)


def backfill(
    client: NarClient, store: Store, manifest: Manifest,
    start_ym: str, end_ym: str, today: date,
    finalize_after_days: int = 45, kind: str = "race",
    on_progress=None, daily_refresh_months: int = 2,
) -> list[FetchOutcome]:
    """全期間のバックフィル。

    3秒間隔は client のトークンバケットが強制する。344か月なら約20分。

    `daily_refresh_months` は `fetch_month`/`should_fetch` へそのまま渡す
    （既定2 = IG-16「日次差分は当月＋前月のみ」を変えない）。一度きりの確定化
    スイープ（`operation/scripts/finalize_history_backlog.py`）だけが、過去分
    すべてを対象に含める大きな値を明示的に渡す。
    """
    out = []
    months = month_range(start_ym, end_ym)
    for i, ym in enumerate(months, start=1):
        res = fetch_month(client, store, manifest, ym, today, finalize_after_days,
                          daily_refresh_months, kind=kind)
        out.append(res)
        if on_progress:
            on_progress(i, len(months), res)
    return out
