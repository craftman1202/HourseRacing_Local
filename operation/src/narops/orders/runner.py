"""PC 上の発注ランナー。

  python -m narops.orders.runner            # 常駐（20 秒ごとに GCS の発注計画を見る）
  python -m narops.orders.runner --once     # 1 回だけ
  touch ~/.cache/nar-orders/STOP            # 緊急停止（次の巡回から何も発注しない）

実発注の条件（全部）: conf/orders.yaml の live: true、環境変数 NAR_ORDERS_LIVE=1、停止ファイルが無い。
どれかが欠ければドライラン。結果は必ず Discord に出す（発注・見送り・エラーのどれも）。
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .driver import Driver, OrderResult
from .guard import KILL_FILE, Limits, decide
from .ledger import Ledger
from .plan import PREFIX, OrderPlan

log = logging.getLogger("orders")
CONF = Path(__file__).resolve().parents[3] / "conf/orders.yaml"


def is_live(limits: Limits) -> bool:
    return bool(limits.live) and os.environ.get("NAR_ORDERS_LIVE") == "1"


def process(plan: OrderPlan, limits: Limits, ledger: Ledger, driver: Driver, report,
            now: datetime, errors: list[int]) -> None:
    """1 レース分。errors は連続エラー数（要素 1 個のリストで共有）。"""
    live = is_live(limits)
    d = decide(plan, limits, now, ledger.spent(plan.race_date, live), ledger.done(plan.race_id))
    if d.reason_all:
        if "窓の前" not in d.reason_all and plan.lines and not ledger.done(plan.race_id):
            report.skip_all(plan, d.reason_all)
            for ln in plan.lines:
                ledger.record(plan.race_id, plan.race_date, ln.bet_type, ln.tickets, ln.stake_yen,
                              "skipped", d.reason_all, now)
        return
    for ln, why in d.skipped:
        if why != "発注済み":
            ledger.record(plan.race_id, plan.race_date, ln.bet_type, ln.tickets, ln.stake_yen,
                          "skipped", why, now)
            report.skipped(plan, ln, why)
    for ln in d.allowed:
        if errors[0] >= limits.max_consecutive_errors:
            why = f"連続エラー {errors[0]} 回で停止中"
            ledger.record(plan.race_id, plan.race_date, ln.bet_type, ln.tickets, ln.stake_yen, "skipped", why, now)
            report.skipped(plan, ln, why)
            continue
        try:
            res = driver.place(plan, ln, submit=live)
        except Exception as exc:  # noqa: BLE001
            res = OrderResult("error", f"{type(exc).__name__}: {str(exc)[:200]}")
        errors[0] = errors[0] + 1 if res.status == "error" else 0
        ledger.record(plan.race_id, plan.race_date, ln.bet_type, ln.tickets, ln.stake_yen,
                      res.status, res.detail, now)
        report.line(plan, ln, res, live)
        if errors[0] >= limits.max_consecutive_errors:
            report.halt(errors[0])


def load_plans(date: str, bucket: str, project: str | None) -> list[OrderPlan]:
    from google.cloud import storage

    client = storage.Client(project=project)
    out = []
    for b in client.list_blobs(bucket, prefix=f"{PREFIX}/{date}/"):
        try:
            out.append(OrderPlan.from_json(b.download_as_text()))
        except Exception as exc:  # noqa: BLE001
            log.warning("発注計画を読めません %s: %s", b.name, exc)
    return sorted(out, key=lambda p: p.start_ts)


def main() -> int:
    from .report import DiscordReporter
    from .rakuten import RakutenDriver          # 楽天の画面操作（実機確認後に実装）

    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--conf", default=str(CONF))
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument("--plan-file", help="GCS の代わりにローカルの発注計画 JSON（1 レース分）を読む（動作確認用）")
    ap.add_argument("--live", action="store_true", help="NAR_ORDERS_LIVE=1 と同じ（設定の live: true も必要）")
    ap.add_argument("--no-discord", action="store_true", help="Discord に送らない（動作確認用）")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from ..config import load_dotenv

    for k, v in load_dotenv(Path(__file__).resolve().parents[3] / ".env").items():
        os.environ.setdefault(k, v)                      # Discord の webhook（.env）。認証情報は出さない
    if a.live:
        os.environ["NAR_ORDERS_LIVE"] = "1"
    limits = Limits.load(a.conf)
    bucket = os.environ.get("NAROPS_ORDER_BUCKET") or "nar-raw-sample-335613"
    ledger, errors = Ledger(), [0]
    report = DiscordReporter(webhook="") if a.no_discord else DiscordReporter()
    if a.no_discord:
        report.sender = None
    driver = RakutenDriver()
    report.started(limits, is_live(limits))
    try:
        while True:
            now = datetime.now(timezone.utc)
            day = now.astimezone().strftime("%Y-%m-%d")
            plans = ([OrderPlan.from_json(Path(a.plan_file).read_text(encoding="utf-8"))] if a.plan_file
                     else load_plans(day, bucket, os.environ.get("NAROPS_PROJECT", "sample-335613")))
            for plan in plans:
                process(plan, limits, ledger, driver, report, now, errors)
            if a.once:
                break
            time.sleep(a.interval)
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
