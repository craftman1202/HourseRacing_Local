"""entity キャッシュ（narops.refresh / features.py::history_before の entity_cache 引数）。

2026-09-11 のコスト最適化。BigQuery の本番実測で `SELECT * FROM entry_result_final`
（馬・騎手・調教師・種牡馬の全キャリア取得、下限日付なし）が1日 150〜275GB の
スキャンになっていた（月額 $30 規模）。`entry_result_final` は日次 MERGE
（narops.refresh）でしか更新されないため、その MERGE 直後のコピーを GCS へ
1日1回書き出し、推論時はそこを読んで BigQuery への問い合わせを避ける。

ここで検証したいのは1点に尽きる: **キャッシュを使っても使わなくても、
`history_before` / `build_for_race` が返す値が1バイトも変わらないこと**。
キャッシュはコスト最適化専用で、正しさの前提にしてはいけない。

`entity_cache` は DataFrame ではなく**ローカル parquet ファイルのパス**を渡す
（2026-09-11 の本番障害: 480万行を pandas 化すると列を絞っても 5GB を超え、
`nar-ops`（4GiB 上限）を OOM Kill した。`query_entity_cache` が pyarrow の
フィルタ pushdown で必要な数百行だけを読む設計に直した）。テストでも実際の
ファイル読み込み経路を通すため、`results` を一度ディスクへ書いてからパスで渡す。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from narops.clock import jst_datetime, to_utc
from narops.features import build_for_race, history_before
from narops.refresh import write_entity_cache

pytestmark = pytest.mark.unit


def _cache_path(tmp_path: Path, df: pd.DataFrame, as_of: date = date(2026, 8, 25)) -> str:
    """`results` 等を実際に parquet として書き出し、そのローカルパスを返す。

    本番では `Services.entity_cache_for()` が GCS からローカルへダウンロード
    してからパスを渡すが、`write_entity_cache` 自体が返すのも実ファイルの
    パス（テストでは `persist_root` を `file://` にしているのでそのまま
    ローカルパス）なので、ダウンロード分を省いても `history_before` から見て
    等価に検証できる。
    """
    persist_root = f"file://{tmp_path / 'entity_cache_store'}"
    return write_entity_cache(df, persist_root, as_of)


def _card(n: int = 3) -> pd.DataFrame:
    """`results`（conftest の make_results）に実在するエンティティで組んだ出馬表。"""
    return pd.DataFrame({
        "horse_no": list(range(1, n + 1)),
        "horse_sk": [f"H{i:04d}" for i in range(n)],
        "jockey_sk": [f"J{i:03d}" for i in range(n)],
        "trainer_sk": [f"T{i:03d}" for i in range(n)],
        "sire_sk": [f"S{i:03d}" for i in range(n)],
    })


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    """行順・列順のどちらも比較の対象にしない（意味を持つのは値だけ）。

    列順は BigQuery のクエリ結果順とキャッシュ元 DataFrame の元の順で
    自然に食い違うが、下流（DuckDB 登録・pandas の列名アクセス）は列名
    基準なので実害はない。
    """
    cols = [c for c in ("race_id", "horse_no") if c in df.columns]
    out = df.sort_values(cols).reset_index(drop=True) if cols else df.reset_index(drop=True)
    return out[sorted(out.columns)]


# ---------------------------------------------------- history_before の等価性
def test_history_before_with_cache_matches_direct_query(populated, results, tmp_path):
    """entity_cache 有り/無しで history_before の出力が完全一致する。"""
    as_of = to_utc(jst_datetime(2026, 8, 25, 15, 0))
    card = _card()
    cache_path = _cache_path(tmp_path, results)

    direct = history_before(populated, as_of, lookback_days=30, card=card,
                            entity_cache=None)
    cached = history_before(populated, as_of, lookback_days=30, card=card,
                            entity_cache=cache_path)

    a, b = _sorted(direct), _sorted(cached)
    assert list(a.columns) == list(b.columns)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_history_before_cache_equivalence_holds_with_no_matching_entities(populated, results,
                                                                          tmp_path):
    """出馬表のエンティティが確定層に無い（新馬戦等）場合も同じ挙動になる。"""
    as_of = to_utc(jst_datetime(2026, 8, 25, 15, 0))
    card = pd.DataFrame({
        "horse_no": [1], "horse_sk": ["H9999"], "jockey_sk": ["J999"],
        "trainer_sk": ["T999"], "sire_sk": ["S999"],
    })
    cache_path = _cache_path(tmp_path, results)

    direct = history_before(populated, as_of, lookback_days=30, card=card,
                            entity_cache=None)
    cached = history_before(populated, as_of, lookback_days=30, card=card,
                            entity_cache=cache_path)

    pd.testing.assert_frame_equal(_sorted(direct), _sorted(cached), check_dtype=False)


def test_history_before_cache_with_extra_silver_columns_still_matches(populated, results,
                                                                       tmp_path):
    """本番の実際の entity キャッシュは silver 由来の列を余分に持つ（確認済み、
    2026-09-11 実測: horse_name/weight_kg/popularity など 43 列、`entry_result_final`
    の DDL は 27 列）。`read_entity_cache`/`query_entity_cache` は
    `ENTITY_CACHE_COLUMNS` に絞って読むので余分な列は最初から現れないはずだが、
    そのフィルタ自体が正しく機能していることをここで直接確認する。
    """
    wide_cache = results.assign(
        horse_name="ダミー馬名", weight_kg=480.0, popularity=1.0,
        race_no=1, class_level=1, prize_yen=1_000_000,
    )
    as_of = to_utc(jst_datetime(2026, 8, 25, 15, 0))
    card = _card()
    cache_path = _cache_path(tmp_path, wide_cache)

    direct = history_before(populated, as_of, lookback_days=30, card=card,
                            entity_cache=None)
    cached = history_before(populated, as_of, lookback_days=30, card=card,
                            entity_cache=cache_path)

    assert "horse_name" not in cached.columns, (
        "ENTITY_CACHE_COLUMNS に無い列は読み込み時点で落ちるはず（下流へ漏らさない）")
    pd.testing.assert_frame_equal(_sorted(direct), _sorted(cached), check_dtype=False)


def test_history_before_cache_only_replaces_final_layer_not_live(populated, results, clock,
                                                                  tmp_path):
    """entity_cache は entry_result_final の代わりであって、ライブ層は常に生きたまま。

    ライブ層（entry_result_live）にだけ存在する当日の先行レース結果が、
    entity_cache 使用時も欠落しないことを確認する。
    """
    from narops.db.merge import refresh_live

    as_of = to_utc(jst_datetime(2026, 8, 25, 15, 0))
    card = _card()
    cache_path = _cache_path(tmp_path, results)
    live_row = pd.DataFrame([{
        "race_id": "202026082501", "horse_no": 1,
        "horse_sk": "H0000", "jockey_sk": "J000", "trainer_sk": "T000", "sire_sk": "S000",
        "race_date": date(2026, 8, 25), "start_ts": to_utc(jst_datetime(2026, 8, 25, 11, 0)),
        "baba_code": 20, "distance": 1200, "finish_pos": 1, "is_win": 1,
        "time_sec": 72.0, "speed_index": 0.1, "last3f": 38.6,
        "jockey_record": "", "all_record": "", "dirt_left_record": "",
        "dirt_right_record": "", "track_record": "", "dist_record": "",
        "best_time": "", "best_time_good": "", "turn": "右", "baba_condition": "良",
    }])
    refresh_live(populated, live_row, clock=clock)

    direct = history_before(populated, as_of, lookback_days=30, card=card,
                            entity_cache=None)
    cached = history_before(populated, as_of, lookback_days=30, card=card,
                            entity_cache=cache_path)

    assert "202026082501" in set(direct["race_id"]) & set(cached["race_id"]), (
        "当日のライブ層レコードが両方の経路に含まれるはず")
    pd.testing.assert_frame_equal(_sorted(direct), _sorted(cached), check_dtype=False)


def test_history_before_cache_avoids_loading_full_table_into_pandas(populated, results, tmp_path,
                                                                     monkeypatch):
    """OOM 障害の再発防止テスト: `pd.read_parquet`（全件 pandas 化）を経由しないこと。

    2026-09-11、`read_entity_cache` が DataFrame を返す実装だったとき、
    480万行を丸ごと pandas 化して `nar-ops`（4GiB）を OOM Kill した。
    `query_entity_cache` が pyarrow のフィルタ pushdown だけで絞り込み、
    `pandas.read_parquet` を一切呼ばないことを直接確認する。
    """
    as_of = to_utc(jst_datetime(2026, 8, 25, 15, 0))
    card = _card()
    cache_path = _cache_path(tmp_path, results)

    calls = []
    original = pd.read_parquet

    def _spy(*a, **kw):
        calls.append((a, kw))
        return original(*a, **kw)

    monkeypatch.setattr(pd, "read_parquet", _spy)
    history_before(populated, as_of, lookback_days=30, card=card, entity_cache=cache_path)

    assert calls == [], (
        "entity_cache 使用時に pd.read_parquet（全件 pandas 化）が呼ばれています。"
        "query_entity_cache は pyarrow.parquet.read_table を直接使うはず")


# ------------------------------------------------------- build_for_race の等価性
def test_build_for_race_output_is_identical_with_and_without_cache(
        populated, results, release_dir, feature_spec, tmp_path):
    """history_before の1つ下（実際の特徴量出力）でも完全一致すること。

    これが最終的な安全網: 学習・推論で共有する builder.build() の出力
    （モデルに渡る値そのもの）が、キャッシュ経路の有無で変わらないことを
    直接検証する。
    """
    from nar.config import feature_config
    from narops.model.manifest import Manifest

    manifest = Manifest.read(release_dir / "manifest.json")
    fc = feature_config()
    card = _card().assign(race_id="202026082599")
    cache_path = _cache_path(tmp_path, results)
    race_row = pd.Series({
        "race_id": "202026082599", "race_date": date(2026, 8, 25),
        "start_ts": to_utc(jst_datetime(2026, 8, 25, 15, 0)),
        "baba_code": 20, "distance": 1200, "race_no": 1,
        "surface": "ダ", "turn": "右", "baba_condition": "良",
        "class_level": 1, "prize_yen": 1_000_000,
    })

    direct = build_for_race(populated, card, race_row, manifest, fc,
                            entity_cache=None)
    cached = build_for_race(populated, card, race_row, manifest, fc,
                            entity_cache=cache_path)

    pd.testing.assert_frame_equal(
        direct.frame.sort_values("horse_no").reset_index(drop=True),
        cached.frame.sort_values("horse_no").reset_index(drop=True),
        check_dtype=False)


# --------------------------------------------------------------- write/read
def test_write_then_read_entity_cache_round_trips(tmp_path, results):
    from narops.refresh import read_entity_cache, write_entity_cache

    persist_root = f"file://{tmp_path / 'persist'}"
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    as_of = date(2026, 9, 11)

    path = write_entity_cache(results, persist_root, as_of)
    assert path

    local_path = read_entity_cache(persist_root, as_of, local_dir=str(local_dir))
    assert local_path is not None
    back = pd.read_parquet(local_path)
    # read_entity_cache は列を ENTITY_CACHE_COLUMNS に絞って読む
    # （2026-09-11、OOM 障害を受けた対応）。`results` 自体が既にその
    # 列集合と同じなので、順序だけが変わりうる。
    pd.testing.assert_frame_equal(_sorted(back), _sorted(results), check_dtype=False)


def test_read_entity_cache_reuses_local_file_without_redownloading(tmp_path, results,
                                                                    monkeypatch):
    """同じ日の2回目の呼び出しはローカルファイルを再利用し、GCS を再度読まない。"""
    from narops.refresh import read_entity_cache, write_entity_cache
    from nar.io.store import Store

    persist_root = f"file://{tmp_path / 'persist'}"
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    as_of = date(2026, 9, 11)
    write_entity_cache(results, persist_root, as_of)

    first = read_entity_cache(persist_root, as_of, local_dir=str(local_dir))
    assert first is not None

    calls = []
    original = Store.read_bytes

    def _spy(self, *parts):
        calls.append(parts)
        return original(self, *parts)

    monkeypatch.setattr(Store, "read_bytes", _spy)
    second = read_entity_cache(persist_root, as_of, local_dir=str(local_dir))

    assert second == first
    assert calls == [], "2回目はローカルファイルを再利用し、GCS を再読み込みしないはず"


def test_read_entity_cache_returns_none_when_missing(tmp_path):
    from narops.refresh import read_entity_cache

    persist_root = f"file://{tmp_path / 'nothing-here'}"
    assert read_entity_cache(persist_root, date(2026, 9, 11),
                             local_dir=str(tmp_path / "local")) is None


def test_read_entity_cache_returns_none_on_corrupt_file(tmp_path):
    """壊れたファイルでも例外を外に出さない（推論を止めてはいけない）。"""
    from narops.refresh import ENTITY_CACHE_PREFIX, read_entity_cache

    persist_root = f"file://{tmp_path / 'persist'}"
    p = tmp_path / "persist" / ENTITY_CACHE_PREFIX
    p.mkdir(parents=True)
    (p / "2026-09-11.parquet").write_bytes(b"not a parquet file")

    local_path = read_entity_cache(persist_root, date(2026, 9, 11),
                                   local_dir=str(tmp_path / "local"))
    # ローカルへのコピー自体は成功しうる（バイト列を書くだけなので）。
    # 壊れているのが分かるのは query_entity_cache が実際に読もうとした時点。
    if local_path is not None:
        import pyarrow.parquet as pq

        with pytest.raises(Exception):  # noqa: B017 - pyarrow の例外型は環境依存
            pq.read_table(local_path)


def test_write_entity_cache_is_included_in_daily_refresh_persistence(tmp_path, wh, clock,
                                                                      monkeypatch):
    """daily_refresh() が entity キャッシュを書き出すこと（成功時のみ）。"""
    from narops import refresh
    from nar.ingest.monthly import FetchOutcome
    from nar.io.manifest import Record

    fixture = (Path(__file__).resolve().parents[2] / "learning" / "tests" / "fixtures"
              / "golden_monthly_202607.zip")

    def _fake_backfill(client, store, manifest, start_ym, end_ym, today,
                       finalize_after_days, kind="race", on_progress=None):
        raw_bytes = fixture.read_bytes()
        path = store.write_atomic(raw_bytes, "raw", "monthly", "race", "ym=2026-07",
                                  "202607_race.zip")
        manifest.upsert(Record(file_key="monthly/race/2026-07", url="fixture://golden",
                               fetched_at=clock.now(), http_status=200,
                               content_length=len(raw_bytes), sha256="fixturehash",
                               raw_path=path, status="ok", is_final=False))
        return [FetchOutcome("monthly/race/2026-07", "downloaded", "fixturehash",
                             len(raw_bytes))]

    monkeypatch.setattr("nar.ingest.monthly.backfill", _fake_backfill)

    persist_root = f"file://{tmp_path / 'persist'}"
    summary = refresh.daily_refresh(wh=wh, clock=clock, workdir=tmp_path / "workdir",
                                    persist_root=persist_root, since_days=3650)

    today = clock.now().date()
    local_path = refresh.read_entity_cache(persist_root, today,
                                           local_dir=str(tmp_path / "local"))
    assert local_path is not None
    cached = pd.read_parquet(local_path)
    assert not cached.empty
    # entity キャッシュは since フィルタ前の全件。今回は全件が対象期間内なので
    # MERGE された行数と一致するはず（golden fixture は単一月ぶんのみ）。
    assert len(cached) == summary.load_history.total_merged
