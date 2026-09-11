"""narops.refresh の日次自動更新パイプラインのテスト。

対応する障害: `/ingest-and-refresh` が `monthly=None` のまま毎日 "0 rows"/"ok" を
記録し続け、確定層が一度も自動更新されず鮮度ゲート（DB-04）が fail-closed に
なっていた（2026-08-29）。ここでは実際の golden fixture ZIP（NAR 実データ）を
使い、`nar ingest → bronze → silver → 確定層 MERGE` の各段の書き換え
（`refresh.run_bronze`/`build_silver_frames`/`load_history`）と、
raw/manifest の GCS 相当ストアへの同期（`sync_dir`）を検証する。
実ネットワークには一切触れない。
"""

from __future__ import annotations

from datetime import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from narops import refresh
from narops.clock import business_date

FIXTURE = (Path(__file__).resolve().parents[2] / "learning" / "tests" / "fixtures"
          / "golden_monthly_202607.zip")


def _seed_store_with_golden_fixture(store) -> "object":
    from nar.io.manifest import Manifest, Record

    manifest = Manifest(store.path("meta", "manifest.duckdb"))
    raw_bytes = FIXTURE.read_bytes()
    path = store.write_atomic(raw_bytes, "raw", "monthly", "race", "ym=2026-07",
                              "202607_race.zip")
    manifest.upsert(Record(file_key="monthly/race/2026-07", url="fixture://golden",
                           fetched_at=dt.now(), http_status=200,
                           content_length=len(raw_bytes), sha256="fixturehash",
                           raw_path=path, status="ok", is_final=False))
    return manifest


# ------------------------------------------------------------------ sync_dir
def test_sync_dir_copies_files_between_file_roots(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "raw" / "monthly").mkdir(parents=True)
    (src / "raw" / "monthly" / "a.zip").write_bytes(b"hello")
    (src / "meta").mkdir()
    (src / "meta" / "manifest.duckdb").write_bytes(b"fake-db-bytes")

    refresh.sync_dir(f"file://{src}", f"file://{dst}", "raw", "meta/manifest.duckdb")

    assert (dst / "raw" / "monthly" / "a.zip").read_bytes() == b"hello"
    assert (dst / "meta" / "manifest.duckdb").read_bytes() == b"fake-db-bytes"


def test_sync_dir_is_a_silent_noop_when_source_path_is_absent(tmp_path):
    """初回実行相当: 永続化先にまだ raw/manifest が無い状態でエラーにしない。"""
    empty_src = tmp_path / "nothing-here"
    dst = tmp_path / "dst"
    refresh.sync_dir(f"file://{empty_src}", f"file://{dst}", "raw", "meta/manifest.duckdb")
    assert not (dst / "raw").exists()
    assert not (dst / "meta").exists()


# ------------------------------------------------------- bronze / silver
def test_run_bronze_and_build_silver_frames_from_golden_fixture(tmp_path):
    from nar.io.store import Store

    store = Store(f"file://{tmp_path}")
    store.ensure_layout()
    manifest = _seed_store_with_golden_fixture(store)

    n_tables = refresh.run_bronze(store, manifest)
    manifest.close()
    assert n_tables > 0

    frames = refresh.build_silver_frames(store)
    assert not frames["race"].empty
    assert not frames["entry"].empty
    assert "race_id" in frames["entry"].columns
    assert "speed_index" not in frames["entry"].columns, (
        "silver 段階では speed_index は未計算のはず（load_history 側で計算する）")
    # 実データにばんえい（1-4）が含まれていること（除外は load_history の責務）
    assert frames["race"]["baba_code"].isin([1, 2, 3, 4]).any()


def test_build_silver_frames_raises_a_clear_error_when_bronze_is_empty(tmp_path):
    from nar.io.store import Store

    store = Store(f"file://{tmp_path}")
    store.ensure_layout()
    with pytest.raises(refresh.EmptyBronzeError, match="bronze が空です"):
        refresh.build_silver_frames(store)


# ---------------------------------------------------- bronze の確定月キャッシュ
def test_run_bronze_skips_extraction_for_already_bronzed_final_months(tmp_path, monkeypatch):
    """is_final かつ bronze 既存の月は ZIP を再展開しない（コスト最適化の核心）。

    2026-08-29 に追加された `nar-refresh` Job は毎日全履歴を bronze から
    作り直しており、これが元の設計書 §6.1 の見積もりに入っていなかった実コストの
    主因だった。確定済み月の再展開が実際に起きていないことを、
    `build_from_zip` の呼び出し回数で直接検証する。
    """
    from nar.io.store import Store
    from nar.io.manifest import Manifest, Record
    from nar.transform import bronze as bz

    store = Store(f"file://{tmp_path}")
    store.ensure_layout()
    manifest = Manifest(store.path("meta", "manifest.duckdb"))
    raw_bytes = FIXTURE.read_bytes()
    path = store.write_atomic(raw_bytes, "raw", "monthly", "race", "ym=2026-07",
                              "202607_race.zip")
    manifest.upsert(Record(file_key="monthly/race/2026-07", url="fixture://golden",
                           fetched_at=dt.now(), http_status=200,
                           content_length=len(raw_bytes), sha256="fixturehash",
                           raw_path=path, status="ok", is_final=True))

    assert refresh.run_bronze(store, manifest) > 0

    calls: list = []
    original = bz.build_from_zip

    def _spy(*a, **kw):
        calls.append(a)
        return original(*a, **kw)

    monkeypatch.setattr(bz, "build_from_zip", _spy)

    n2 = refresh.run_bronze(store, manifest)
    manifest.close()

    assert n2 == 0
    assert calls == [], "is_final かつ bronze 既存の月は ZIP を再展開しないはず"

    # スキップしても既存の bronze データ自体はそのまま読める
    frames = refresh.build_silver_frames(store)
    assert not frames["race"].empty


def test_run_bronze_still_reprocesses_non_final_months_every_time(tmp_path):
    """is_final=False（未確定・直近）の月は bronze が既にあっても毎回展開し直す。

    NAR 側の事後訂正（降着・失格など、設計書 §2.1 第三の経路）を拾うため、
    確定前の月にキャッシュを効かせてはいけない。
    """
    from nar.io.store import Store
    from nar.io.manifest import Manifest, Record

    store = Store(f"file://{tmp_path}")
    store.ensure_layout()
    manifest = Manifest(store.path("meta", "manifest.duckdb"))
    raw_bytes = FIXTURE.read_bytes()
    path = store.write_atomic(raw_bytes, "raw", "monthly", "race", "ym=2026-07",
                              "202607_race.zip")
    manifest.upsert(Record(file_key="monthly/race/2026-07", url="fixture://golden",
                           fetched_at=dt.now(), http_status=200,
                           content_length=len(raw_bytes), sha256="fixturehash",
                           raw_path=path, status="ok", is_final=False))

    assert refresh.run_bronze(store, manifest) > 0
    n2 = refresh.run_bronze(store, manifest)
    manifest.close()

    assert n2 > 0, "未確定の月は bronze が既にあっても再展開されるはず"


# ---------------------------------------------------------------- load_history
@pytest.fixture(scope="module")
def golden_silver_frames() -> dict[str, pd.DataFrame]:
    """golden fixture から実際に silver を組み立てる（モジュール内で1回だけ）。"""
    import tempfile

    from nar.io.store import Store

    with tempfile.TemporaryDirectory() as td:
        store = Store(f"file://{td}")
        store.ensure_layout()
        manifest = _seed_store_with_golden_fixture(store)
        refresh.run_bronze(store, manifest)
        manifest.close()
        return refresh.build_silver_frames(store)


def test_load_history_keeps_banei_so_it_has_past_form(wh, clock, golden_silver_frames):
    """golden fixture は6レース全てばんえい（帯広ば, baba_code=3）。確定層に入ること。

    以前はここで全件落としていた。その結果ばんえいの推論は過去走を1行も引けず、
    毎回 InsufficientData で失敗していた（2026-08-29, race_id=032026082906）。
    平地の特徴量に混ざらないことは、ビルダー側の exclude が保証する
    （test_transform.py::test_tr11_banei_excluded_from_training_set）。
    """
    entry = golden_silver_frames["entry"]
    race = golden_silver_frames["race"]
    assert (entry["baba_code"] == 3).all(), "フィクスチャ前提が変わっています（要テスト見直し）"

    result = refresh.load_history(wh, entry, race, since=None,
                                  source_sha256="deadbeef", clock=clock)

    assert result.total_merged == len(entry)
    stored = wh.table("entry_result_final")
    assert not stored.empty
    assert (stored["baba_code"] == 3).all()


def _with_synthetic_non_banei_race(entry: pd.DataFrame, race: pd.DataFrame,
                                   race_date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """実データ1行を複製し、場コードとレースIDだけ非ばんえい場に差し替える。

    speed_index などが要求する列を漏れなく満たすため、ゼロから作らず実データを
    複製する（実データの型・欠損パターンをそのまま引き継げる）。
    """
    synth_race = race.iloc[[0]].copy()
    synth_race["baba_code"] = 20  # 大井
    synth_race["race_id"] = "202026070199"
    synth_race["race_date"] = race_date

    synth_entries = entry[entry["race_id"] == race.iloc[0]["race_id"]].copy()
    synth_entries["baba_code"] = 20
    synth_entries["race_id"] = "202026070199"
    synth_entries["race_date"] = race_date

    return (pd.concat([entry, synth_entries], ignore_index=True),
           pd.concat([race, synth_race], ignore_index=True))


def test_load_history_merges_both_codes_and_since_filters_by_date(wh, clock,
                                                                  golden_silver_frames):
    """平地・ばんえいのどちらも確定層に入り、--since で日付フィルタされること。"""
    import datetime as _dt

    entry, race = _with_synthetic_non_banei_race(
        golden_silver_frames["entry"], golden_silver_frames["race"],
        race_date=_dt.date(2026, 8, 1))

    # since より前（実フィクスチャの2026-07分＝ばんえい）と since 以降
    # （合成した平地の行）が混在する状態で、日付フィルタが効くことを確認する。
    # 場コードでの選別はもう行わない（ばんえいも確定層に入れる）。
    result = refresh.load_history(wh, entry, race, since=_dt.date(2026, 8, 1),
                                  source_sha256="cafef00d", clock=clock)

    final = wh.table("entry_result_final")
    assert not final.empty
    assert result.total_merged == len(final)
    assert (pd.to_datetime(final["race_date"]).dt.date >= _dt.date(2026, 8, 1)).all(), (
        "since より前の行が確定層に入っています"
    )
    assert final["baba_code"].eq(20).all(), (
        "since より後にあるのは合成した大井の行だけのはず")
    assert final["source_sha256"].eq("cafef00d").all()


# -------------------------------------------------------------- daily_refresh
def _fake_backfill(client, store, manifest, start_ym, end_ym, today,
                   finalize_after_days, kind="race", on_progress=None):
    """実ネットワークの代わりに golden fixture を「取得済み」として manifest に積む。"""
    from nar.ingest.monthly import FetchOutcome
    from nar.io.manifest import Record

    raw_bytes = FIXTURE.read_bytes()
    path = store.write_atomic(raw_bytes, "raw", "monthly", "race", "ym=2026-07",
                              "202607_race.zip")
    manifest.upsert(Record(file_key="monthly/race/2026-07", url="fixture://golden",
                           fetched_at=dt.now(), http_status=200,
                           content_length=len(raw_bytes), sha256="fixturehash",
                           raw_path=path, status="ok", is_final=False))
    return [FetchOutcome("monthly/race/2026-07", "downloaded", "fixturehash", len(raw_bytes))]


def test_daily_refresh_merges_and_persists_state_on_success(tmp_path, wh, clock, monkeypatch):
    monkeypatch.setattr("nar.ingest.monthly.backfill", _fake_backfill)

    persist_root = f"file://{tmp_path / 'persist'}"
    workdir = tmp_path / "workdir"

    summary = refresh.daily_refresh(wh=wh, clock=clock, workdir=workdir,
                                    persist_root=persist_root, since_days=3650)

    assert summary.n_bronze_tables > 0
    # ingest→bronze→silver→merge の全経路がエラー無く完走し、状態が永続化される
    # ことを検証する。golden fixture は全件ばんえいだが、確定層には入る。
    assert summary.load_history.total_merged > 0
    assert wh.table("entry_result_final")["baba_code"].eq(3).all()
    assert (tmp_path / "persist" / "meta" / "manifest.duckdb").exists(), (
        "成功時は manifest を永続化先へ書き戻すはず")
    assert list((tmp_path / "persist" / "raw").rglob("*.zip")), (
        "成功時は raw ZIP を永続化先へ書き戻すはず")
    assert list((tmp_path / "persist" / "bronze").rglob("*.parquet")), (
        "成功時は bronze も永続化先へ書き戻すはず（確定月キャッシュの前提）")


def test_daily_refresh_second_run_reuses_persisted_manifest(tmp_path, wh, clock, monkeypatch):
    """2回目の実行は永続化された manifest/raw を読み込んでから始まること。"""
    monkeypatch.setattr("nar.ingest.monthly.backfill", _fake_backfill)
    persist_root = f"file://{tmp_path / 'persist'}"

    refresh.daily_refresh(wh=wh, clock=clock, workdir=tmp_path / "workdir1",
                          persist_root=persist_root, since_days=3650)
    # 2回目は別の ephemeral workdir（Cloud Run Job の別実行を模す）でも、
    # 永続化先から manifest/raw を復元できること
    summary2 = refresh.daily_refresh(wh=wh, clock=clock, workdir=tmp_path / "workdir2",
                                     persist_root=persist_root, since_days=3650)
    assert summary2.load_history.total_merged >= 0  # 冪等 MERGE なので再投入は「不変」でもよい
    assert (tmp_path / "workdir2" / "raw" / "monthly" / "race" / "ym=2026-07"
            / "202607_race.zip").exists(), "永続化先から raw ZIP が復元されていません"


def test_daily_refresh_does_not_persist_state_on_failure(tmp_path, wh, clock, monkeypatch):
    """失敗時は不完全な状態を永続化先に書き戻さない（次回まっさらから再試行できる）。"""
    monkeypatch.setattr("nar.ingest.monthly.backfill", _fake_backfill)

    def _boom(store):
        raise RuntimeError("silver 構築が壊れたシミュレーション")

    monkeypatch.setattr(refresh, "build_silver_frames", _boom)

    persist_root = f"file://{tmp_path / 'persist'}"
    with pytest.raises(RuntimeError, match="壊れたシミュレーション"):
        refresh.daily_refresh(wh=wh, clock=clock, workdir=tmp_path / "workdir",
                              persist_root=persist_root)

    assert not (tmp_path / "persist" / "meta" / "manifest.duckdb").exists(), (
        "失敗時に manifest を永続化先へ書き戻してはいけません")
    assert not (tmp_path / "persist" / "bronze").exists(), (
        "失敗時に bronze を永続化先へ書き戻してはいけません")


def test_main_requires_ingest_store_env_var(monkeypatch):
    """NAROPS_INGEST_STORE 未設定なら黙って動かさず即座に落ちる（DC-02）。"""
    monkeypatch.delenv("NAROPS_INGEST_STORE", raising=False)
    with pytest.raises(RuntimeError, match="NAROPS_INGEST_STORE"):
        refresh.main()
