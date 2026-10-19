"""PC 上の発注ランナー。

  python -m narops.orders.runner            # 常駐（20 秒ごとに GCS の発注計画を見る）
  python -m narops.orders.runner --once     # 1 回だけ
  touch ~/.cache/nar-orders/STOP            # 一時停止（全ランナー。静かに待つだけ。消せば再開）
  touch ~/.cache/nar-orders/STOP.<名前>     # その名前のランナーだけ一時停止

複数のランナー（地方競馬向けの別サービスなど）が同時に動いてよい。二重発注は台帳の原子的な宣言
（ledger.claim）で防ぐ。1 日の損失上限は同じ楽天アカウントの残高で測るので、台帳を共有する。

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


def _skip_new_lines(plan: OrderPlan, ledger: Ledger, done, now, why: str) -> None:
    for ln in plan.lines:
        if (plan.race_id, ln.bet_type) not in done:
            ledger.record(plan.race_id, plan.race_date, ln.bet_type, ln.tickets, ln.stake_yen, "skipped", why, now)


def process(plan: OrderPlan, limits: Limits, ledger: Ledger, driver: Driver, report,
            now: datetime, errors: list[int], kill_files: tuple[Path, ...] = (KILL_FILE,)) -> None:
    """1 レース分。errors は連続エラー数（要素 1 個のリストで共有）。

    1 日の上限は損失（guard.py の docstring）。損失 = その日の基準残高 − 今の精算可能額 + 処理中の額。
    基準残高はその日の最初に記録する（途中から動き始めたときは 今の残高 + 今日すでに受け付けた額。
    払戻が既にあれば損失を多めに見積もる側＝安全側になる）。残高を読めなければ発注しない。
    """
    live = is_live(limits)
    done = ledger.done(plan.race_id)
    d = decide(plan, limits, now, 0, done, kill_files)
    if d.reason_all:                       # 窓の前・締切後・一時停止・発走後の古い計画
        if not d.quiet and plan.lines and not done:
            report.skip_all(plan, d.reason_all)
            _skip_new_lines(plan, ledger, done, now, d.reason_all)
        return
    if all((plan.race_id, ln.bet_type) in done for ln in plan.lines):
        return                             # 新しい買い目が無い（残高を読みに行かない）
    try:
        bal, why_bal = driver.balance(plan), ""
    except Exception as exc:  # noqa: BLE001
        bal, why_bal = None, f"{type(exc).__name__}: {str(exc)[:120]}"
    if bal is None:
        why = "精算可能額を読めず、1日の損失上限を確認できないため見送り" + (f"（{why_bal}）" if why_bal else "")
        errors[0] += 1
        report.skip_all(plan, why)
        _skip_new_lines(plan, ledger, done, now, why)
        if errors[0] >= limits.max_consecutive_errors:
            report.halt(errors[0])
        return
    base = ledger.baseline(plan.race_date)
    if base is None:
        base = bal + ledger.stake_total(plan.race_date, ("placed",))
        ledger.set_baseline(plan.race_date, base, now)
    loss = base - bal + ledger.stake_total(plan.race_date, ("pending",))
    d = decide(plan, limits, now, loss, done, kill_files)
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
        if not ledger.claim(plan.race_id, plan.race_date, ln.bet_type, ln.tickets, ln.stake_yen, now):
            continue                                   # 別のランナーが先に処理している（二重発注の防止）
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
    ap.add_argument("--conf", default=os.environ.get("NAR_ORDERS_CONF") or str(CONF))
    ap.add_argument("--name", default=os.environ.get("NAR_ORDERS_NAME") or "main",
                    help="このランナーの名前（systemd のインスタンス名。通知の表示と個別の一時停止に使う）")
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument("--plan-file", help="GCS の代わりにローカルの発注計画 JSON（1 レース分）を読む（動作確認用）")
    ap.add_argument("--live", action="store_true", help="NAR_ORDERS_LIVE=1 と同じ（設定の live: true も必要）")
    ap.add_argument("--ledger", help="発注台帳（sqlite）の場所。中央競馬用など、ランナーを分けるときに別にする")
    ap.add_argument("--no-discord", action="store_true", help="Discord に送らない（動作確認用）")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    # httpx は INFO で Discord の webhook URL（秘密）をそのままログに出す。出さない
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    from ..config import load_dotenv

    for k, v in load_dotenv(Path(__file__).resolve().parents[3] / ".env").items():
        os.environ.setdefault(k, v)                      # Discord の webhook（.env）。認証情報は出さない
    if a.live:
        os.environ["NAR_ORDERS_LIVE"] = "1"
    limits = Limits.load(a.conf)
    bucket = os.environ.get("NAROPS_ORDER_BUCKET") or "nar-raw-sample-335613"
    ledger, errors = Ledger(a.ledger), [0]
    from .guard import STATE_DIR

    kill_files = (KILL_FILE, STATE_DIR / f"STOP.{a.name}")
    report = DiscordReporter(webhook="", name=a.name) if a.no_discord else DiscordReporter(name=a.name)
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
                process(plan, limits, ledger, driver, report, now, errors, kill_files)
            if a.once:
                break
            time.sleep(a.interval)
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
