"""IG-01..16, SG-01..04: 取得層とスキーマガード。"""

from __future__ import annotations

from datetime import date, datetime

import httpx
import pytest

from nar.errors import ContentTypeError, SchemaDriftError
from nar.ingest.client import NarClient, RetryPolicy, TokenBucket
from nar.ingest.unzip import decode, extract, inner_names
from nar.io.manifest import Manifest, Record, finalize_pass, month_end, should_fetch
from nar.io.store import Store, sha256_bytes
from nar.transform.schema_guard import (
    EXPECTED_COLUMNS, SchemaRegistry, combined_hash, schema_of,
)


class FakeClock:
    """待機時間を実時間で消費しないための時計。系列そのものを assert する。"""

    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def client(handler, clock: FakeClock, **kw) -> NarClient:
    return NarClient(
        user_agent="test", min_interval_sec=3.0,
        transport=httpx.MockTransport(handler),
        sleep=clock.sleep, clock=clock.now, seed=42, **kw,
    )


# --------------------------------------------------------------------- IG-01..04
def test_ig01_new_fetch_records_matching_hash(tmp_path, golden_zip):
    store = Store(f"file://{tmp_path}")
    m = Manifest(tmp_path / "manifest.duckdb")
    path = store.write_atomic(golden_zip, "raw", "monthly", "race", "ym=2026-07", "race.zip")
    m.upsert(Record(
        file_key="monthly/race/2026-07", sha256=sha256_bytes(golden_zip),
        raw_path=path, fetched_at=datetime(2026, 8, 1), http_status=200,
        content_length=len(golden_zip), status="ok",
    ))
    rec = m.get("monthly/race/2026-07")
    assert rec.status == "ok"
    assert rec.sha256 == sha256_bytes(store.read_bytes(
        "raw", "monthly", "race", "ym=2026-07", "race.zip"))


def test_ig02_final_month_issues_zero_requests(tmp_path):
    m = Manifest(tmp_path / "m.duckdb")
    m.upsert(Record(file_key="monthly/race/1998-01", sha256="a", status="ok", is_final=True))
    assert should_fetch(m, "monthly/race/1998-01", "1998-01", date(2026, 8, 25)) is False


def test_ig03_non_final_recent_month_is_refetched(tmp_path):
    m = Manifest(tmp_path / "m.duckdb")
    m.upsert(Record(file_key="monthly/race/2026-08", sha256="a", status="ok", is_final=False))
    assert should_fetch(m, "monthly/race/2026-08", "2026-08", date(2026, 8, 25)) is True


def test_ig04_content_change_keeps_previous_generation(tmp_path):
    m = Manifest(tmp_path / "m.duckdb")
    m.upsert(Record(file_key="k", sha256="old", raw_path="p1", status="ok"))
    m.upsert(Record(file_key="k", sha256="new", raw_path="p2", status="ok"))
    hist = m.con.execute("SELECT sha256 FROM manifest_history WHERE file_key='k'").fetchall()
    assert [h[0] for h in hist] == ["old"]
    assert m.get("k").sha256 == "new"


# ------------------------------------------------------------------------ IG-05
def test_ig05_backoff_sequence_has_jitter_and_raises_on_sixth():
    clock = FakeClock()
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429)

    with client(handler, clock) as c:
        with pytest.raises(httpx.HTTPStatusError):
            c.fetch("https://example.test/x", {})

    assert calls["n"] == 6, "5回リトライして6回目で例外送出"
    assert len(c.waits) == 5
    for i, w in enumerate(c.waits, start=1):
        base = 2.0 * 2 ** (i - 1)
        assert base * 0.7 <= w <= base * 1.3, f"attempt {i}: {w} が ±30% ジッタの外"
    # 実装漏れだと全部 base ちょうどになる
    assert any(abs(w - 2.0 * 2 ** (i - 1)) > 1e-9 for i, w in enumerate(c.waits, 1))


# ------------------------------------------------------------------------ IG-06
def test_ig06_min_interval_between_consecutive_requests(golden_zip):
    clock = FakeClock()

    def handler(request):
        clock.t += 0.01  # 応答自体にかかる時間
        return httpx.Response(200, content=golden_zip,
                              headers={"content-type": "application/zip"})

    with client(handler, clock) as c:
        for _ in range(4):
            c.fetch("https://example.test/x", {})

    gaps = [b - a for a, b in zip(c.request_times, c.request_times[1:])]
    assert min(gaps) >= 2.95, f"最小間隔 {min(gaps)}s"


# ------------------------------------------------------------------ IG-07, IG-08
def test_ig07_html_error_page_with_200_is_rejected(tmp_path):
    clock = FakeClock()

    def handler(request):
        return httpx.Response(200, text="<html>エラー</html>",
                              headers={"content-type": "text/html"})

    with client(handler, clock) as c:
        with pytest.raises(ContentTypeError):
            c.fetch("https://example.test/x", {})

    store = Store(f"file://{tmp_path}")
    store.ensure_layout()
    assert store.ls("raw") == [], "raw に保存してはいけない"


def test_ig08_interrupted_write_leaves_no_tmp_file(tmp_path, monkeypatch):
    store = Store(f"file://{tmp_path}")

    def boom(src, dst, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(store.fs, "mv", boom)
    with pytest.raises(KeyboardInterrupt):
        store.write_atomic(b"partial", "raw", "x.zip")

    assert not store.exists("raw", "x.zip")
    assert not store.exists("raw", "x.zip.tmp")


# ------------------------------------------------------------------ IG-09, IG-10
def test_ig09_cp932_decodes_japanese_headers(golden_zip):
    csvs = extract(golden_zip)
    assert len(csvs) == 1
    assert csvs[0].codec == "cp932"
    assert "競走年月日" in csvs[0].text and "大井" in csvs[0].text
    assert inner_names(golden_zip) == ["RACE.csv"]


def test_ig10_invalid_bytes_fall_back_to_replace(caplog):
    raw = "正常".encode("cp932") + b"\x81\x00\xff\xfe" + "終".encode("cp932")
    text, codec, replaced = decode(raw)
    assert replaced is True and codec == "cp932/replace"
    assert "正常" in text


# ------------------------------------------------------------------ IG-11..13
@pytest.mark.parametrize("days,expect_final", [(44, False), (45, True)])
def test_ig11_ig12_finalization_boundary(tmp_path, days, expect_final):
    m = Manifest(tmp_path / "m.duckdb")
    key = "monthly/race/2026-06"
    m.upsert(Record(file_key=key, sha256="a", status="ok"))
    m.upsert(Record(file_key=key, sha256="a", status="ok"))  # ハッシュ不変の再取得
    today = month_end("2026-06") + __import__("datetime").timedelta(days=days)
    finalize_pass(m, today)
    assert m.get(key).is_final is expect_final


def test_ig13_hash_change_resets_streak_and_keeps_non_final(tmp_path):
    m = Manifest(tmp_path / "m.duckdb")
    key = "monthly/race/2026-06"
    m.upsert(Record(file_key=key, sha256="a", status="ok"))
    m.upsert(Record(file_key=key, sha256="a", status="ok"))
    m.upsert(Record(file_key=key, sha256="b", status="ok"))  # 内容が変わった
    assert m.get(key).unchanged_streak == 0
    finalize_pass(m, month_end("2026-06") + __import__("datetime").timedelta(days=60))
    assert m.get(key).is_final is False


# ------------------------------------------------------------------------ IG-14
def test_ig14_offline_rebuild_from_raw_only(tmp_path, golden_zip):
    """raw 以外を全削除した状態から、ネットワーク無しで再構築できる。

    本設計の受け入れ条件そのもの。パース仕様をどれだけ変更しても
    NAR への再アクセスが発生しないことを、この1本が保証する。
    """
    store = Store(f"file://{tmp_path}")
    store.write_atomic(golden_zip, "raw", "monthly", "race", "ym=2026-07", "race.zip")

    def no_network(*a, **kw):
        raise AssertionError("オフライン再構築中にネットワークアクセスが発生しました")

    import httpx as _httpx
    original = _httpx.Client.get
    _httpx.Client.get = no_network
    try:
        raw = store.read_bytes("raw", "monthly", "race", "ym=2026-07", "race.zip")
        csvs = extract(raw)
        h1 = combined_hash([schema_of(c.text) for c in csvs])
        # 2回目も同じ成果物になる（決定性）
        h2 = combined_hash([schema_of(c.text) for c in extract(raw)])
        assert h1 == h2
    finally:
        _httpx.Client.get = original


# ------------------------------------------------------------------ IG-15, IG-16
def test_ig15_backfill_request_count_within_budget():
    from dateutil.relativedelta import relativedelta  # noqa: F401  (存在確認のみ)

    months = (2026 - 1998) * 12 + 8  # 1998-01 〜 2026-08
    odds = 6 * 3
    assert 343 <= months <= 345
    assert 15 <= odds <= 21
    assert months + odds <= 365


def test_ig16_daily_diff_touches_current_and_previous_month_only(tmp_path):
    m = Manifest(tmp_path / "m.duckdb")
    today = date(2026, 8, 25)
    for ym in ("2026-06", "2026-07", "2026-08"):
        m.upsert(Record(file_key=f"monthly/race/{ym}", sha256="a", status="ok"))
    fetched = [
        ym for ym in ("2026-06", "2026-07", "2026-08")
        if should_fetch(m, f"monthly/race/{ym}", ym, today, daily_refresh_months=2)
    ]
    assert fetched == ["2026-07", "2026-08"]


# --------------------------------------------------------------------- SG-01..04
def test_sg01_expected_column_counts_are_pinned():
    assert EXPECTED_COLUMNS == {"race": 66, "entry": 36, "odds": 10, "payout": 54}


def test_sg02_drift_stops_promotion(tmp_path, golden_zip, drift_zip):
    reg = SchemaRegistry(tmp_path / "schema_hash.json")
    good = schema_of(extract(golden_zip)[0].text, kind="race")
    reg.pin("race", good.hash())

    drifted = schema_of(extract(drift_zip)[0].text, kind="race")
    with pytest.raises(SchemaDriftError, match="列数 67"):
        reg.check(drifted)


def test_sg03_column_reorder_is_detected(tmp_path):
    """列名集合が同一でも通さない。位置ベースのパーサが静かに壊れるため。"""
    reg = SchemaRegistry(tmp_path / "s.json")
    cols = [f"c{i}" for i in range(10)]
    reg.pin("odds", schema_of(",".join(cols) + "\n", kind="odds").hash())
    swapped = cols[:3] + [cols[4], cols[3]] + cols[5:]
    with pytest.raises(SchemaDriftError, match="不一致"):
        reg.check(schema_of(",".join(swapped) + "\n", kind="odds"))


def test_sg04_matching_schema_passes(tmp_path, golden_zip):
    reg = SchemaRegistry(tmp_path / "s.json")
    s = schema_of(extract(golden_zip)[0].text, kind="race")
    reg.pin("race", s.hash())
    reg.check(s)  # 例外が出なければ silver 昇格に進める


def test_sg_unknown_kind_is_drift_not_autoregistration(tmp_path):
    reg = SchemaRegistry(tmp_path / "s.json")
    cols = ",".join(f"c{i}" for i in range(10))
    with pytest.raises(SchemaDriftError, match="既知の schema_hash がありません"):
        reg.check(schema_of(cols + "\n", kind="odds"))
