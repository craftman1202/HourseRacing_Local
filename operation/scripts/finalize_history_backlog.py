"""bronze の確定月キャッシュを実際に効かせるための一度きりの確定化スイープ。

## なぜ必要か

`nar-refresh` Job（2026-09-10 のコスト最適化）は `is_final` な月の bronze
再展開をスキップする（`operation/src/narops/refresh.py`）。しかし `is_final` は
「月末+45日経過」かつ「直近の再取得でハッシュが前回と同じだったことを実際に
観測した」の両方が揃って初めて `nar.io.manifest.finalize_pass` が立てる
（IG-12/13）。

日次の自動更新（`narops.refresh.daily_refresh`）は `should_fetch()` の設計
（IG-16: 当月＋前月のみ再取得）で、それ以外の月には一切触れない。初回の
一括バックフィルで入った過去分（1998年〜、約340か月）は、一度取得された
直後にこの「当月＋前月」の窓の外に落ちるため、**二度目の「不変の観測」が
永久に発生せず、`unchanged_streak` が初期値0のまま進まない**。結果として
`finalize_pass` の条件（`unchanged_streak >= 1`）を満たせず、`is_final` が
一度も True にならない（2026-09-10、本番 manifest で実際に確認した状態）。

このスクリプトは `daily_refresh_months` を全履歴をカバーする値まで広げて
`backfill()` を一度だけ呼び、過去分すべてに「不変の確認」を1回与える。
これで `finalize_pass` が `is_final=True` にでき、以降の日次実行で
bronze の確定月キャッシュが実際に効くようになる。日々の自動更新自体の
挙動（IG-16、当月＋前月限定）はこのスクリプトでは変更しない。

## 実行コスト

対象月数 × `min_interval_sec`（既定3秒）が概算所要時間。1998年〜現在で
約345か月なら17〜20分。NAR の本番サイトへ実際にリクエストするので、
何度も繰り返し実行するものではない（一度 `is_final` になった月は、以降
このスクリプトを含めどこからも再取得されなくなる）。

## 使い方

    cd operation
    export PYTHONPATH=src:../learning/src
    export NAROPS_INGEST_STORE=gs://nar-raw-sample-335613/ingest-store
    python scripts/finalize_history_backlog.py            # 計画のみ（何も取得しない）
    python scripts/finalize_history_backlog.py --apply     # 実行

`--apply` を付けない限り何も取得しない。既定は対象月数・未確定件数・想定所要
時間の表示だけ。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("finalize_history_backlog")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="bronze の確定月キャッシュを効かせるための一度きりの確定化スイープ")
    p.add_argument("--apply", action="store_true", help="実際に NAR へ再取得し、manifest を更新する")
    p.add_argument("--workdir", default=os.environ.get(
        "NAROPS_FINALIZE_WORKDIR", "/tmp/nar-finalize-workdir"))
    args = p.parse_args(argv)

    persist_root = os.environ.get("NAROPS_INGEST_STORE")
    if not persist_root:
        print("NAROPS_INGEST_STORE が未設定です（raw/manifest の永続化先の GCS URI が必要です）",
              file=sys.stderr)
        return 1

    from nar.config import data_config
    from nar.ingest.client import NarClient, RetryPolicy
    from nar.ingest.monthly import backfill, month_range
    from nar.io.manifest import Manifest, finalize_pass
    from nar.io.store import Store
    from narops.clock import SystemClock, business_date
    from narops.refresh import sync_dir

    workdir = Path(args.workdir)
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
    start_ym = dcfg["backfill"]["race_start_ym"]
    finalize_after_days = int(dcfg["backfill"]["finalize_after_days"])
    clock = SystemClock()
    today = business_date(clock.now())
    end_ym = f"{today.year:04d}-{today.month:02d}"

    months = month_range(start_ym, end_ym)
    all_before = manifest.all()
    not_final_before = sum(1 for r in all_before if r.status == "ok" and not r.is_final)
    final_keys_before = {r.file_key for r in all_before if r.is_final}
    est_minutes = len(months) * float(http["min_interval_sec"]) / 60
    print(f"対象月数 {len(months)}（{start_ym} 〜 {end_ym}）、うち未確定 {not_final_before}")
    print(f"想定所要時間: 約 {est_minutes:.0f} 分（min_interval_sec={http['min_interval_sec']}）")

    if not args.apply:
        print("計画のみ。実行するには --apply を付けてください。")
        manifest.close()
        return 0

    client = NarClient(
        user_agent=http["user_agent"], min_interval_sec=float(http["min_interval_sec"]),
        timeout_sec=float(http["timeout_sec"]),
        retry=RetryPolicy(statuses=tuple(http["retry"]["statuses"]),
                          max_attempts=int(http["retry"]["max_attempts"]),
                          base_sec=float(http["retry"]["base_sec"]),
                          factor=float(http["retry"]["factor"]),
                          jitter=float(http["retry"]["jitter"])))

    def _progress(i: int, n: int, res) -> None:
        if i % 20 == 0 or i == n:
            log.info("%d/%d %s: %s", i, n, res.file_key, res.status)

    try:
        backfill(client, store, manifest, start_ym, end_ym, today,
                 finalize_after_days, kind="race", on_progress=_progress,
                 daily_refresh_months=len(months) + 1)
    finally:
        client.close()

    # `fetch_month` の「不変」分岐は upsert 時点で直接 is_final を立てる
    # （`nar.io.manifest.fetch_month` 参照）。`finalize_pass` はその後の
    # 保険（streak が別経路で溜まっていた場合の取りこぼし救済）に過ぎず、
    # その戻り値だけを見ると「今回の確定件数」を大きく過小報告する。
    # 実際に確定した件数は before/after の is_final 集合の差分で数える。
    finalize_pass(manifest, today, finalize_after_days)
    final_keys_after = {r.file_key for r in manifest.all() if r.is_final}
    manifest.close()

    log.info("永続化状態を %s へ書き戻します", persist_root)
    sync_dir(local_root, persist_root, "raw", "bronze",
            "meta/manifest.duckdb", "meta/track_master.json")

    n_finalized = len(final_keys_after - final_keys_before)
    print(f"完了。今回新たに確定した月: {n_finalized}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
