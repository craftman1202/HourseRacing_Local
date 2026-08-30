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
"""

from __future__ import annotations

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
    """
    import re

    from nar.transform import bronze as bz

    n_tables = 0
    for rec in manifest.all():
        if rec.status != "ok":
            continue
        m = re.search(r"\d{4}-\d{2}", rec.file_key)
        if m is None:
            log.warning("%s: 年月を判定できません", rec.file_key)
            continue
        ym = m.group(0)
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


def load_history(wh: Any, entry: pd.DataFrame, race: pd.DataFrame, *,
                 since: date | None, source_sha256: str = "",
                 clock: Clock | None = None,
                 reason: str = "monthly-backfill") -> LoadHistoryResult:
    """確定層への投入。`cli.py::cmd_load_history` と全く同じロジック（挙動を変えない
    ようリファクタしただけ）。CLI と `daily_refresh()` の両方から使う。

    speed_index は必ず引数の `entry` 全体（フィルタ前）で計算し、`since` は
    その後の行フィルタにしか使わない（`--since` は「投入する行」を絞るだけ）。
    """
    from nar.features.builder import speed_index

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
    entry = entry.copy()
    # 着順が無い行（取消・除外の馬、中止レース、未実施の予定）も残す。
    # 学習側のウィンドウ集計は silver の全行を見ており、確定層だけ間引くと
    # 出走数がずれる。「まだ結果が来ていない」ことは finish_pos が NULL で
    # 表現され、鮮度ゲートは着順が入った行だけを見るので、予定を最新と誤認しない。
    n_pending = int(entry["finish_pos"].isna().sum())

    attrs = [c for c in ("turn", "baba_condition") if c in race.columns]
    if attrs:
        lookup = race.drop_duplicates("race_id").set_index("race_id")
        for c in attrs:
            entry[c] = entry["race_id"].map(lookup[c])

    entry = speed_index(entry)
    if since is not None:
        since_ts = pd.Timestamp(since)
        span = pd.to_datetime(entry["race_date"])
        entry = entry[span >= since_ts]

    ascii_names = {v: k for k, v in RECORD_COLUMN_MAP.items()}
    payload = entry.rename(columns=ascii_names)
    # silver の start_ts は NAR ファイル由来の JST の naive。DB 内の naive は
    # 常に UTC という規約（DB-10）なので、そのまま入れると 9 時間ずれる。
    payload["start_ts"] = (pd.to_datetime(payload["start_ts"])
                           .dt.tz_localize(JST).dt.tz_convert("UTC").dt.tz_localize(None))

    total = 0
    per_year: list[tuple[int, MergeResult]] = []
    payload["_year"] = pd.to_datetime(payload["race_date"]).dt.year
    for year, chunk in payload.groupby("_year", sort=True):
        res = merge_final(wh, chunk.drop(columns=["_year"]),
                          source_sha256=source_sha256, clock=clock, reason=reason)
        total += res.inserted + res.updated
        per_year.append((int(year), res))

    freshness = check_freshness(wh, clock)
    return LoadHistoryResult(total, n_pending, per_year, freshness)


# ------------------------------------------------------------ daily refresh
@dataclass
class RefreshSummary:
    n_fetched: int
    n_bronze_tables: int
    load_history: LoadHistoryResult


def daily_refresh(*, wh: Any, clock: Clock, workdir: Path, persist_root: str,
                  since_days: int = 7) -> RefreshSummary:
    """ingest → bronze → silver → 確定層 MERGE の全経路。

    `persist_root`（GCS 想定）は raw ZIP と manifest だけを永続化する。成功時のみ
    書き戻す（失敗した回の不完全な状態を次回に持ち越さない — 次回は素直に
    もう一度取得し直す）。bronze/silver はモジュール docstring の理由で毎回
    ローカルの `workdir` に作り直す。
    """
    from nar.config import data_config
    from nar.ingest.client import NarClient, RetryPolicy
    from nar.ingest.monthly import backfill
    from nar.io.manifest import Manifest, finalize_pass
    from nar.io.store import Store

    workdir.mkdir(parents=True, exist_ok=True)
    local_root = f"file://{workdir}"

    log.info("永続化状態を %s から同期します", persist_root)
    sync_dir(persist_root, local_root, "raw", "meta/manifest.duckdb", "meta/track_master.json")

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

    log.info("永続化状態を %s へ書き戻します", persist_root)
    sync_dir(local_root, persist_root, "raw", "meta/manifest.duckdb", "meta/track_master.json")

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
