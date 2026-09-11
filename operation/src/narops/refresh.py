"""確定層の自動日次更新。

`/ingest-and-refresh`（`service.py::ingest_and_refresh_endpoint`）はこれまで
`monthly` を渡されたことが無く、確定層への MERGE が一度も自動実行されて
いなかった（毎回 "0 rows" で "ok" と記録され、鮮度ゲート DB-04 が
fail-closed になっていた実際の障害・2026-08-29）。ここでは
`nar ingest → nar bronze → nar silver → 確定層 MERGE` の全経路を Cloud Run Job
（`python -m narops.refresh`）として毎日実行できるようにする。

speed_index は場×距離の as-of 統計を使うため、**必ず全履歴で計算する**
（`cli.py::cmd_load_history` と同じ制約。直近月だけで計算すると基準統計量が
MIN_PRIOR_FOR_SPEED_INDEX に届かずほぼ全行 NaN になる）。

GCS 上に直接 bronze/silver を作らない理由: `nar.transform.bronze`/`silver` は
内部で `pathlib.Path` を直接使っており（`bronze.read`/`write`/`available_months`
など）、fsspec 抽象を通していない。`Store` 自体は raw ZIP バイト列の
読み書き（`write_atomic`/`read_bytes`）だけ gs:// 対応で、bronze/silver 層は
file:// 専用というのが実際のコードの姿（設計書が謳う「DATA_ROOT を gs:// に
差し替えるだけで移行できる」は raw 層にしか当てはまらない）。このギャップは
`learning/src/nar` 側の既存コードを変更しないと直せず、今回のタスクの範囲を
超えるため、**raw ZIP と manifest だけを GCS に永続化し、bronze/silver は
毎回ローカルの ephemeral ワーキングディレクトリで作り直す**ことで回避する。

#### bronze の確定月キャッシュ（コスト最適化、2026-09-10）

上の設計のまま運用すると、`run_bronze` が manifest 上の**全月**（1998年〜）を
毎日 ZIP 展開からやり直す。これが `nar-refresh` Job を 16Gi・4vCPU・タイムアウト
3600s まで太らせた主因で（2026-08-28 時点の設計書 §6.1 のコスト見積もりは
この Job の存在自体を含んでおらず、実運用コストが試算から丸ごと漏れていた）、
Cloud Run の無料枠（450,000 GiB秒/月）を日次実行だけで大きく超える可能性がある。

`raw` の再取得が `is_final`（月末から45日経過＋直近チェックでハッシュ不変を
確認済み）を条件に止まっているのと同じ理由で、**確定済み月の bronze 再展開も
情報を増やさない**。そこで bronze を raw/manifest と同様に `persist_root` へ
永続化し（`sync_dir` の対象に追加）、`run_bronze` は「`is_final` かつ bronze が
既に存在する月」を展開せずスキップする。speed_index の as-of 基準統計量は
従来どおり**全履歴の bronze から**計算するので（`build_silver_frames` は
変更しない）、推論品質・特徴量の値は一切変わらない。減るのは「変化しないと
分かっている過去分を毎日 CSV から再デコードする」という純粋な無駄な計算だけ。

#### entity キャッシュ（コスト最適化、2026-09-11）

`operation/src/narops/features.py::history_before` は、馬・騎手・調教師・種牡馬
それぞれの**全キャリア**を BigQuery `entry_result_final` から取得する（下限日付を
付けられない — デビュー日が不明な設計上の制約）。この「下限なし」クエリを
**レースごとに最大5本**（4エンティティ種別 × 確定層・ライブ層）発行しており、
1日約65レースぶん積み重なると、本番実測で1日 150〜275GB のスキャンになっていた
（2026-09-11、`INFORMATION_SCHEMA.JOBS_BY_PROJECT` で確認）。無料枠 1TiB/月を
毎日単独で消費するペースで、月額 $30 規模の実コストになっていた。

`entry_result_final` は本モジュールのこの日次 MERGE でしか更新されない
（当日の途中経過はライブ層 `entry_result_live` が別頻度で持つ）。したがって
「今朝この MERGE が使った全件データ」は、次の MERGE（明日の同じ時刻）まで
`entry_result_final` に対する `SELECT *` と**厳密に同じ内容であり続ける** —
近似ではなく数学的に保証された等価性。この事実を使い、`load_history()` が
確定層へ投入する直前のデータ（`LoadHistoryResult.full_payload`）を
そのまま GCS（`persist_root/entity_cache/{当日}.parquet`）へ書き出す
（`write_entity_cache`）。`history_before` はレースごとに BigQuery へ
問い合わせる代わりに、まずこのファイルを読んで pandas 側でエンティティを
絞り込む。BigQuery のクエリ課金（バイト数）を経由しないので、レース数に
比例していた課金がほぼゼロになる。

集計ロジック（`nar.features.builder.build`）は一切変更していない。変わるのは
「同じ生データ行を BigQuery から毎回取るか、1日1回作ったコピーから読むか」
という**データの取得経路だけ**で、渡される行の内容は同一である。ファイルが
無い場合（初回デプロイ直後・当日分がまだ無い・前日の実行が失敗した）は
`history_before` が黙って直接 BigQuery へフォールバックする。キャッシュは
純粋な高速化・コスト最適化であり、正しさの前提にはしない。
"""

from __future__ import annotations

import io
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .clock import JST, Clock, SystemClock, business_date
from .config import OpsConfig, SecretResolver
from .db.freshness import Freshness
from .db.freshness import check as check_freshness
from .db.merge import MergeResult, merge_final
from .db.schema import RECORD_COLUMN_MAP
from .jobs import record_run

log = logging.getLogger("narops.refresh")


# ------------------------------------------------------------------- sync
def sync_dir(src_root: str, dst_root: str, *relative_paths: str) -> None:
    """`src_root` 配下の指定パスを `dst_root` へコピーする。

    fsspec 経由なので file:// と gs:// のどちらでも同じコードで動く
    （テストは file:// 同士で「GCS 相当」の永続化往復を検証する）。
    コピー元にパスが存在しなければ黙ってスキップする（初回実行時は
    raw/manifest がまだ無いのが正常なので、それ自体はエラーにしない）。
    """
    import fsspec

    src_fs, src_base = fsspec.core.url_to_fs(src_root)
    dst_fs, dst_base = fsspec.core.url_to_fs(dst_root)
    src_base = src_base.rstrip("/")
    dst_base = dst_base.rstrip("/")
    for rel in relative_paths:
        src_path = f"{src_base}/{rel}"
        if not src_fs.exists(src_path):
            continue
        files = src_fs.find(src_path) if src_fs.isdir(src_path) else [src_path]
        for f in files:
            rel_f = f[len(src_base) + 1:]
            dst_path = f"{dst_base}/{rel_f}"
            dst_fs.makedirs(dst_path.rsplit("/", 1)[0], exist_ok=True)
            with src_fs.open(f, "rb") as r, dst_fs.open(dst_path, "wb") as w:
                w.write(r.read())


# ----------------------------------------------------------------- bronze
def run_bronze(store: Any, manifest: Any) -> int:
    """raw の ZIP を展開・デコードして bronze に置く。`cli.py::cmd_bronze` と同じ処理。

    `rec.raw_path` を直接は使わない。マニフェストは GCS で永続化して別プロセス
    （別コンテナ実行・別マシン）から読むため、書き込み時の store の絶対パスが
    そのまま入っている `raw_path` は別 workdir では存在しない
    （実際にこれで最初の Cloud Run Job 実行が全件失敗した — 2026-08-29）。
    `file_key`（`monthly/race/{ym}`）から現在の `store` 基準でパスを組み直す。

    `is_final` な月かつ bronze（race・entry の両方）が既に存在する月は展開を
    スキップする（モジュール docstring「bronze の確定月キャッシュ」参照）。
    `daily_refresh` が `persist_root` から bronze を同期してくるので、2回目
    以降の実行では対象が「当月＋直近の未確定分」だけに絞られる。
    """
    import re

    from nar.transform import bronze as bz

    already_bronzed = (set(bz.available_months(store, "race"))
                       & set(bz.available_months(store, "entry")))
    n_tables = 0
    for rec in manifest.all():
        if rec.status != "ok":
            continue
        m = re.search(r"\d{4}-\d{2}", rec.file_key)
        if m is None:
            log.warning("%s: 年月を判定できません", rec.file_key)
            continue
        ym = m.group(0)
        if rec.is_final and ym in already_bronzed:
            continue
        try:
            raw = store.read_bytes("raw", "monthly", "race", f"ym={ym}",
                                   f"{ym.replace('-', '')}_race.zip")
        except OSError as exc:
            log.warning("%s: raw を読めません (%s)", rec.file_key, exc)
            continue
        for table in bz.build_from_zip(raw, ym, source=rec.file_key):
            bz.write(store, table)
            n_tables += 1
    return n_tables


# ----------------------------------------------------------------- silver
class EmptyBronzeError(RuntimeError):
    pass


def build_silver_frames(store: Any) -> dict[str, pd.DataFrame]:
    """bronze から race/entry/payout/odds の DataFrame を組み立てる（cmd_silver と同じ）。

    必ず全履歴（store にある全 ym）を対象にする — speed_index の as-of
    基準統計量がここでの母集団に依存するため、範囲を絞ってはいけない。
    """
    from nar.transform import bronze as bz
    from nar.transform import silver as sv
    from nar.transform.keys import TrackMaster

    master = TrackMaster.load(store)
    races, entries, payouts, oddss = [], [], [], []
    months = bz.available_months(store, "race")
    for ym in months:
        r_raw, e_raw, p_raw = (bz.read(store, k, ym) for k in ("race", "entry", "payout"))
        if r_raw.empty or e_raw.empty:
            continue
        try:
            race = sv.build_race(r_raw, master)
            entry = sv.attach_start_ts(sv.build_entry(e_raw, master), race)
            races.append(race)
            entries.append(entry)
            if not p_raw.empty:
                payouts.append(sv.build_payout(p_raw, master))
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: silver 構築に失敗 (%s)", ym, exc)
            continue
        o_raw = bz.read(store, "odds", ym)
        if not o_raw.empty:
            try:
                oddss.append(sv.build_odds(o_raw, master))
            except Exception as exc:  # noqa: BLE001
                log.warning("%s: odds 構築に失敗 (%s)", ym, exc)

    if not races:
        raise EmptyBronzeError(
            "bronze が空です。先に `nar ingest` と `nar bronze` を実行してください。")

    out = {"race": pd.concat(races, ignore_index=True).drop_duplicates("race_id"),
           "entry": pd.concat(entries, ignore_index=True)
                      .drop_duplicates(["race_id", "horse_no"]),
           "payout": (pd.concat(payouts, ignore_index=True) if payouts
                      else pd.DataFrame())}
    out["odds"] = (pd.concat(oddss, ignore_index=True)
                     .drop_duplicates(["race_id", "horse_no"]) if oddss
                   else pd.DataFrame())
    return out


# ------------------------------------------------------------ load_history
@dataclass
class LoadHistoryResult:
    total_merged: int
    n_pending: int
    per_year: list[tuple[int, MergeResult]]
    freshness: Freshness
    full_payload: pd.DataFrame


def _to_bq_payload(entry: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
    """silver の `entry`（日本語ヘッダ）を確定層の ASCII スキーマへ変換する。

    `load_history()` の中核変換だけを切り出したもの。`daily_refresh()` の
    entity キャッシュ書き出し（`write_entity_cache`）も同じ変換を要するため、
    2箇所に書くと将来どちらかだけ直してスキューする。1箇所にしてある。
    """
    from nar.features.builder import speed_index

    entry = entry.copy()
    attrs = [c for c in ("turn", "baba_condition") if c in race.columns]
    if attrs:
        lookup = race.drop_duplicates("race_id").set_index("race_id")
        for c in attrs:
            entry[c] = entry["race_id"].map(lookup[c])

    # 上がり3F は silver では文字列（空文字が欠測）。DDL は DOUBLE なので、ここで
    # 明示的に数値化する。DuckDB は文字列を黙って受けるが BigQuery は型で落ちるため、
    # ローカルだけ通って本番で落ちる典型パターンになる。
    if "last3f" in entry.columns:
        entry["last3f"] = pd.to_numeric(entry["last3f"], errors="coerce")

    entry = speed_index(entry)

    ascii_names = {v: k for k, v in RECORD_COLUMN_MAP.items()}
    payload = entry.rename(columns=ascii_names)
    # silver の start_ts は NAR ファイル由来の JST の naive。DB 内の naive は
    # 常に UTC という規約（DB-10）なので、そのまま入れると 9 時間ずれる。
    payload["start_ts"] = (pd.to_datetime(payload["start_ts"])
                           .dt.tz_localize(JST).dt.tz_convert("UTC").dt.tz_localize(None))
    return payload


def load_history(wh: Any, entry: pd.DataFrame, race: pd.DataFrame, *,
                 since: date | None, source_sha256: str = "",
                 clock: Clock | None = None,
                 reason: str = "monthly-backfill") -> LoadHistoryResult:
    """確定層への投入。`cli.py::cmd_load_history` と全く同じロジック（挙動を変えない
    ようリファクタしただけ）。CLI と `daily_refresh()` の両方から使う。

    speed_index は必ず引数の `entry` 全体（フィルタ前）で計算し、`since` は
    その後の行フィルタにしか使わない（`--since` は「投入する行」を絞るだけ）。

    `LoadHistoryResult.full_payload` は `since` フィルタを掛ける**前**の全件
    （ASCII スキーマ変換済み）。`daily_refresh()` がこれを entity キャッシュ
    （`write_entity_cache`）としてそのまま GCS へ書き出す — 確定層に実際に
    MERGE される値と**同一の変換結果**なので、キャッシュと確定層が食い違う
    余地がない。
    """
    clock = clock or SystemClock()
    # ばんえいもここに入れる。以前は投入前に落としていたが、それだと確定層に
    # 過去走が1行も無く、ばんえいの推論は必ず InsufficientData で失敗していた
    # （2026-08-29、race_id=032026082906）。
    #
    # 平地の特徴量に混ざる心配は無い。混入の経路は2つしかなく、どちらも塞がっている:
    #   1. 特徴量ビルダーは conf/features.yaml の exclude で 1-4 を必ず落とす
    #   2. speed_index は (baba_code, distance) ごとに標準化するので、
    #      ばんえい（1-4 × 200m）は平地とグループを共有しない
    # 逆にばんえい側は conf_banei/features.yaml の include で 1-4 だけを取る。
    #
    # 着順が無い行（取消・除外の馬、中止レース、未実施の予定）も残す。
    # 学習側のウィンドウ集計は silver の全行を見ており、確定層だけ間引くと
    # 出走数がずれる。「まだ結果が来ていない」ことは finish_pos が NULL で
    # 表現され、鮮度ゲートは着順が入った行だけを見るので、予定を最新と誤認しない。
    n_pending = int(entry["finish_pos"].isna().sum())

    full_payload = _to_bq_payload(entry, race)

    payload = full_payload
    if since is not None:
        since_ts = pd.Timestamp(since)
        span = pd.to_datetime(payload["race_date"])
        payload = payload[span >= since_ts]

    total = 0
    per_year: list[tuple[int, MergeResult]] = []
    payload = payload.copy()
    payload["_year"] = pd.to_datetime(payload["race_date"]).dt.year
    for year, chunk in payload.groupby("_year", sort=True):
        res = merge_final(wh, chunk.drop(columns=["_year"]),
                          source_sha256=source_sha256, clock=clock, reason=reason)
        total += res.inserted + res.updated
        per_year.append((int(year), res))

    freshness = check_freshness(wh, clock)
    return LoadHistoryResult(total, n_pending, per_year, freshness, full_payload)


# ------------------------------------------------------------ entity cache
ENTITY_CACHE_PREFIX = "entity_cache"


def _entity_cache_name(as_of: date) -> str:
    return f"{as_of.isoformat()}.parquet"


def write_entity_cache(full_payload: pd.DataFrame, persist_root: str, as_of: date) -> str:
    """確定層と同一の全件データを GCS へ1日1回だけ書き出す。

    モジュール docstring「entity キャッシュ」節の実体。`full_payload` は
    `load_history()` が確定層へ MERGE する直前のデータで、`entry_result_final`
    に実際に入る値と1バイトも違わない。日付ごとに別ファイルにする
    （`{as_of}.parquet`）ので、GCS のライフサイクルで数日後に削除しても
    当日分の欠落にはならない（`deploy.py::LIFECYCLE` 参照）。
    """
    from nar.io.store import Store

    store = Store(persist_root)
    buf = io.BytesIO()
    full_payload.to_parquet(buf, index=False)
    return store.write_atomic(buf.getvalue(), ENTITY_CACHE_PREFIX, _entity_cache_name(as_of))


# `entry_result_final` の DDL 列（source_sha256/merged_at は merge_final が
# 後付けする列で full_payload には無い）。full_payload（silver 由来）は
# horse_name/weight_kg のような特徴量計算に使わない列も多数持つため
# （2026-09-11 の本番障害で判明: 43列 × 480万行を丸ごとロードして nar-ops が
# 4096MiB を超え OOM Kill された）、キャッシュを読む際はこの列だけに絞る。
ENTITY_CACHE_COLUMNS = (
    "race_id", "horse_no", "horse_sk", "jockey_sk", "trainer_sk", "sire_sk",
    "race_date", "start_ts", "baba_code", "distance", "finish_pos", "is_win",
    "time_sec", "speed_index", "jockey_record", "all_record", "dirt_left_record",
    "dirt_right_record", "track_record", "dist_record", "best_time",
    "best_time_good", "turn", "baba_condition", "last3f",
)


def read_entity_cache(persist_root: str, as_of: date, local_dir: str = "/tmp") -> str | None:
    """`write_entity_cache` が書いた当日分を**ローカルへ**取得し、パスを返す。

    `nar-ops`（`/infer`）側から呼ぶ。無ければ（未デプロイ直後・その日の
    `nar-refresh` がまだ／失敗）**例外を投げず None を返す**。呼び出し側
    （`features.py::history_before`）は None を「キャッシュ無し」として直接
    BigQuery へフォールバックする設計なので、ここで落とすと安いはずの経路が
    そのまま推論停止になってしまう。キャッシュは高速化・コスト最適化のみが
    目的で、可用性の前提にしない。

    **DataFrame を返さない理由**（2026-09-11 の本番障害、実測で判明）:
    480万行 × 25列でも `pd.read_parquet()` で全件 pandas 化すると
    メモリ上で 5GB を超え、`nar-ops`（4GiB 上限）を単独で OOM Kill する
    （列を絞っても、行数そのものが支配的でほぼ効かなかった）。
    代わりにファイルをローカルディスクへ保存するだけに留め、実際の絞り込みは
    `query_entity_cache` が `pyarrow` のフィルタ pushdown で行毎に読む
    （実測: 同じファイルに対する1エンティティぶんの絞り込み読み出しが
    0.6秒・1MB未満）。ファイル自体は 274MB 程度で、ローカルへ置くだけなら
    安全な増分。
    """
    from nar.io.store import Store

    name = _entity_cache_name(as_of)
    local_path = f"{local_dir.rstrip('/')}/narops-entity-cache-{as_of.isoformat()}.parquet"
    if os.path.exists(local_path):
        return local_path
    try:
        store = Store(persist_root)
        if not store.exists(ENTITY_CACHE_PREFIX, name):
            return None
        data = store.read_bytes(ENTITY_CACHE_PREFIX, name)
        os.makedirs(local_dir, exist_ok=True)
        tmp = f"{local_path}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, local_path)
        return local_path
    except Exception as exc:  # noqa: BLE001
        log.warning("entity キャッシュを読めません（%s）。直接 BigQuery へフォールバックします",
                   exc)
        return None


def query_entity_cache(local_path: str, col: str, values: list[str],
                       upto: date) -> pd.DataFrame:
    """ローカルの entity キャッシュから、指定エンティティの全キャリアだけを
    `pyarrow` のフィルタ pushdown で読む。

    `pd.read_parquet()` で全件を pandas 化してから `.loc[mask]` で絞る
    実装は、480万行が丸ごとメモリに載ってしまい OOM を起こした
    （2026-09-11、本番障害）。`pyarrow.parquet.read_table(..., filters=...)`
    は Arrow のネイティブ表現のまま絞り込んでから pandas 化するため、
    最終的にメモリに残るのは実際に一致した数十〜数百行だけになる。
    """
    import pyarrow.parquet as pq

    table = pq.read_table(local_path, columns=list(ENTITY_CACHE_COLUMNS),
                          filters=[(col, "in", values), ("race_date", "<=", upto)])
    return table.to_pandas()


# ------------------------------------------------------------ daily refresh
@dataclass
class RefreshSummary:
    n_fetched: int
    n_bronze_tables: int
    load_history: LoadHistoryResult


def daily_refresh(*, wh: Any, clock: Clock, workdir: Path, persist_root: str,
                  since_days: int = 7) -> RefreshSummary:
    """ingest → bronze → silver → 確定層 MERGE の全経路。

    `persist_root`（GCS 想定）は raw ZIP・確定月の bronze・manifest を永続化する。
    成功時のみ書き戻す（失敗した回の不完全な状態を次回に持ち越さない — 次回は
    素直にもう一度取得し直す）。silver はモジュール docstring の理由（bronze/silver
    が fsspec 未対応）で毎回ローカルの `workdir` から全履歴の bronze を読んで
    作り直す。bronze 自体は「bronze の確定月キャッシュ」節の通り、確定済み月は
    `run_bronze` が再展開をスキップするので、実際に ZIP から作り直すのは
    当月＋直近の未確定分だけになる。
    """
    from nar.config import data_config
    from nar.ingest.client import NarClient, RetryPolicy
    from nar.ingest.monthly import backfill
    from nar.io.manifest import Manifest, finalize_pass
    from nar.io.store import Store

    workdir.mkdir(parents=True, exist_ok=True)
    local_root = f"file://{workdir}"

    log.info("永続化状態を %s から同期します", persist_root)
    sync_dir(persist_root, local_root, "raw", "bronze",
            "meta/manifest.duckdb", "meta/track_master.json")

    store = Store(local_root)
    store.ensure_layout()
    manifest = Manifest(store.path("meta", "manifest.duckdb"))

    dcfg = data_config()
    http = dcfg["http"]
    today = business_date(clock.now())
    client = NarClient(
        user_agent=http["user_agent"], min_interval_sec=float(http["min_interval_sec"]),
        timeout_sec=float(http["timeout_sec"]),
        retry=RetryPolicy(statuses=tuple(http["retry"]["statuses"]),
                          max_attempts=int(http["retry"]["max_attempts"]),
                          base_sec=float(http["retry"]["base_sec"]),
                          factor=float(http["retry"]["factor"]),
                          jitter=float(http["retry"]["jitter"])))
    try:
        end_ym = f"{today.year:04d}-{today.month:02d}"
        fetched = backfill(client, store, manifest,
                           dcfg["backfill"]["race_start_ym"], end_ym, today,
                           int(dcfg["backfill"]["finalize_after_days"]), kind="race")
    finally:
        client.close()
    finalize_pass(manifest, today, int(dcfg["backfill"]["finalize_after_days"]))

    n_tables = run_bronze(store, manifest)
    dataset_version = manifest.dataset_version()
    manifest.close()

    frames = build_silver_frames(store)
    since = today - timedelta(days=since_days)
    result = load_history(wh, frames["entry"], frames["race"], since=since,
                          source_sha256=dataset_version, clock=clock,
                          reason="daily-refresh")

    cache_path = write_entity_cache(result.full_payload, persist_root, today)
    log.info("entity キャッシュを書き出しました: %s", cache_path)

    log.info("永続化状態を %s へ書き戻します", persist_root)
    sync_dir(local_root, persist_root, "raw", "bronze",
            "meta/manifest.duckdb", "meta/track_master.json")

    return RefreshSummary(len(fetched), n_tables, result)


# --------------------------------------------------------------- entrypoint
def _build_alert_sender(cfg: OpsConfig):
    """`app.py::_attach_discord` と同じ解決順（.env → 環境変数 → Secret Manager）。"""
    from .discord.client import DiscordSender, RateLimiter

    project = os.environ.get("NAROPS_PROJECT") or cfg.project
    sm = None
    if project:
        try:
            from .gcp import SecretManagerResolver

            sm = SecretManagerResolver(project=project)
        except Exception as exc:  # noqa: BLE001
            log.warning("Secret Manager を使えません（%s）。.env と環境変数のみで解決します", exc)
    resolver = SecretResolver(os.environ.get("NAROPS_ENV_FILE", ".env"),
                              secret_manager=sm, project=project)
    alert_url = resolver.get("DISCORD_WEBHOOK_ALERT", required=False)
    if not alert_url:
        return None
    return DiscordSender(alert_url, rate_limiter=RateLimiter(rps=cfg.discord_rate_limit_rps))


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    cfg = OpsConfig.load()
    clock = SystemClock()
    workdir = Path(os.environ.get("NAROPS_REFRESH_WORKDIR", "/tmp/nar-ingest-workdir"))
    persist_root = os.environ.get("NAROPS_INGEST_STORE")
    if not persist_root:
        raise RuntimeError(
            "NAROPS_INGEST_STORE が未設定です（raw/manifest の永続化先の GCS URI が"
            "必要です）。未設定のまま黙って動かしません（DC-02）。")

    from .gcp import BigQueryWarehouse

    wh = BigQueryWarehouse(project=cfg.project,
                           dataset=cfg.raw["gcp"]["resources"]["bq_dataset"],
                           location=cfg.region, max_bytes_billed=cfg.max_bytes_billed)
    alert_sender = _build_alert_sender(cfg)

    try:
        summary = daily_refresh(wh=wh, clock=clock, workdir=workdir,
                                persist_root=persist_root)
    except Exception as exc:  # noqa: BLE001
        msg = f"日次データ更新に失敗しました（fail-closed のまま）: {exc}"
        log.error(msg)
        record_run(wh, "daily_refresh", clock, "failed", str(exc)[:200])
        if alert_sender is not None:
            from .discord.format import alert_embed

            alert_sender.send([alert_embed("fetch_failure", msg, "Blocker")])
        return 1

    detail = (f"{summary.load_history.total_merged:,} 行 merge / "
             f"{summary.load_history.freshness.describe()}")
    record_run(wh, "daily_refresh", clock, "ok", detail)
    log.info("daily_refresh 完了: %s", detail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
