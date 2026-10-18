"""自動発注の安全装置（2026-10-18）。実際の楽天の画面操作は含まない（FakeDriver）。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from narops.orders.driver import FakeDriver
from narops.orders.guard import Limits, decide
from narops.orders.ledger import Ledger
from narops.orders.plan import OrderLine, OrderPlan, from_bet_plans
from narops.orders.runner import is_live, process
from narops.shared import BetPlan

pytestmark = pytest.mark.unit
START = datetime(2026, 10, 18, 6, 30, tzinfo=timezone.utc)


def _plan(lines=None):
    lines = lines or [OrderLine("複勝", ((4,),), 100), OrderLine("3連複", ((2, 5, 7),), 100)]
    return OrderPlan("212026101805", "2026-10-18", START.isoformat(), "川崎", 5, "v", lines)


def _limits(**kw):
    base = dict(live=False, daily_cap_yen=1000, race_cap_yen=500, line_cap_yen=300,
                bet_types=("複勝", "3連複"))
    base.update(kw)
    return Limits(**base)


class Report:
    def __init__(self): self.log = []
    def line(self, plan, ln, res, live): self.log.append(("line", ln.bet_type, res.status))
    def skipped(self, plan, ln, why): self.log.append(("skip", ln.bet_type, why))
    def skip_all(self, plan, why): self.log.append(("skip_all", why))
    def halt(self, n): self.log.append(("halt", n))


def _run(plan, limits, tmp_path, now, driver=None, errors=None, ledger=None):
    led = ledger or Ledger(tmp_path / "l.sqlite")
    drv, rep = driver or FakeDriver(), Report()
    process(plan, limits, led, drv, rep, now, errors or [0])
    return led, drv, rep


def test_json_roundtrip():
    p = _plan()
    assert OrderPlan.from_json(p.to_json()) == p


def test_from_bet_plans_keeps_only_staked_lines():
    a = BetPlan("3連複", "x", 4, [2, 5], 0.4, [(2, 4, 5)], 100, stake=True)
    b = BetPlan("枠連", "y", 4, [2], 0.4, [(2, 3)], 100, stake=False)
    p = from_bet_plans([a, b], race_id="R", race_date="2026-10-18", start_ts=START,
                       track_name="川崎", race_no=5, release="v")
    assert [l.bet_type for l in p.lines] == ["3連複"] and p.lines[0].tickets == ((2, 4, 5),)


def test_default_is_dry_run_and_never_submits(tmp_path, monkeypatch):
    monkeypatch.delenv("NAR_ORDERS_LIVE", raising=False)
    assert not is_live(_limits(live=True))            # live: true だけでは実発注にならない
    _, drv, rep = _run(_plan(), _limits(), tmp_path, START - timedelta(minutes=8))
    assert all(c[2] is False for c in drv.calls) and len(drv.calls) == 2


def test_live_needs_both_switches(tmp_path, monkeypatch):
    monkeypatch.setenv("NAR_ORDERS_LIVE", "1")
    assert is_live(_limits(live=True)) and not is_live(_limits(live=False))
    _, drv, _ = _run(_plan(), _limits(live=True), tmp_path, START - timedelta(minutes=8))
    assert all(c[2] is True for c in drv.calls)


def test_time_window(tmp_path):
    for minutes, expect in [(12, 0), (8, 2), (2, 0)]:
        _, drv, _ = _run(_plan(), _limits(), tmp_path / str(minutes), START - timedelta(minutes=minutes)) \
            if (tmp_path / str(minutes)).mkdir() is None else (None, None, None)
        assert len(drv.calls) == expect, minutes


def test_no_double_order(tmp_path):
    led, drv, _ = _run(_plan(), _limits(), tmp_path, START - timedelta(minutes=8))
    _, drv2, _ = _run(_plan(), _limits(), tmp_path, START - timedelta(minutes=7), ledger=led)
    assert len(drv.calls) == 2 and drv2.calls == []


def test_caps_skip_instead_of_trimming(tmp_path):
    big = [OrderLine("複勝", ((4,),), 100), OrderLine("3連複", tuple((i, i + 1, i + 2) for i in range(1, 6)), 100)]
    _, drv, rep = _run(_plan(big), _limits(line_cap_yen=300), tmp_path, START - timedelta(minutes=8))
    assert [c[1] for c in drv.calls] == ["複勝"]
    assert any(r[0] == "skip" and "上限" in r[2] for r in rep.log)


def test_daily_cap_counts_across_races(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    led.record("other", "2026-10-18", "複勝", ((1,),), 900, "dry_run", "", START)
    _, drv, rep = _run(_plan(), _limits(daily_cap_yen=1000), tmp_path, START - timedelta(minutes=8), ledger=led)
    assert [c[1] for c in drv.calls] == ["複勝"]            # 3連複は 900+100+100 > 1000


def test_bet_type_not_allowed_is_skipped(tmp_path):
    _, drv, _ = _run(_plan(), _limits(bet_types=("複勝",)), tmp_path, START - timedelta(minutes=8))
    assert [c[1] for c in drv.calls] == ["複勝"]


def test_kill_file_stops_everything(tmp_path):
    kill = tmp_path / "STOP"; kill.write_text("x")
    d = decide(_plan(), _limits(), START - timedelta(minutes=8), 0, set(), kill_file=kill)
    assert d.reason_all and not d.allowed


def test_consecutive_errors_halt(tmp_path):
    lines = [OrderLine("複勝", ((4,),), 100), OrderLine("3連複", ((2, 5, 7),), 100)]
    _, drv, rep = _run(_plan(lines), _limits(max_consecutive_errors=2), tmp_path, START - timedelta(minutes=8),
                       driver=FakeDriver(fail_on={"複勝", "3連複"}))
    assert ("halt", 2) in rep.log and len(drv.calls) == 2


def test_rakuten_verify_accepts_exact_match_and_normalizes_order():
    from narops.orders.rakuten import verify
    line = OrderLine("3連複", ((8, 5, 7), (1, 2, 3)), 100)
    rows = [{"horse": "5-7-8", "amount_yen": 100, "race_no": 4, "track": 31, "error": "-"},
            {"horse": "1-2-3", "amount_yen": 100, "race_no": 4, "track": 31, "error": "-"}]
    assert verify(rows, line, 4, 31) is None


def test_rakuten_verify_keeps_order_for_exacta_and_trifecta():
    from narops.orders.rakuten import verify
    line = OrderLine("3連単", ((5, 7, 8),), 200)
    ok = [{"horse": "5→7→8", "amount_yen": 200, "race_no": 4, "track": 31, "error": ""}]
    bad = [{"horse": "5→8→7", "amount_yen": 200, "race_no": 4, "track": 31, "error": ""}]
    assert verify(ok, line, 4) is None and "買い目" in verify(bad, line, 4)


@pytest.mark.parametrize("mutate,expect", [
    (lambda r: r.update(amount_yen=200), "買い目または金額"),
    (lambda r: r.update(race_no=5), "レースが違います"),
    (lambda r: r.update(track=30), "競馬場が違います"),
    (lambda r: r.update(error="発売されていません"), "エラー"),
])
def test_rakuten_verify_rejects_any_difference(mutate, expect):
    from narops.orders.rakuten import verify
    row = {"horse": "5", "amount_yen": 100, "race_no": 4, "track": 31, "error": "-"}
    mutate(row)
    assert expect in verify([row], OrderLine("単勝", ((5,),), 100), 4, 31)


def test_rakuten_verify_rejects_leftover_cart_rows():
    """カゴに前回の買い目が残っていたら件数が合わず止まる。"""
    from narops.orders.rakuten import verify
    rows = [{"horse": "5", "amount_yen": 100, "race_no": 4, "track": 31, "error": ""},
            {"horse": "7", "amount_yen": 100, "race_no": 3, "track": 31, "error": ""}]
    assert "件数" in verify(rows, OrderLine("単勝", ((5,),), 100), 4)


def test_rakuten_driver_refuses_waku_and_late_orders_without_opening_a_browser():
    from narops.orders.rakuten import RakutenDriver
    d = RakutenDriver()
    waku = OrderLine("枠連", ((2, 3),), 100)
    assert d.place(_plan([waku]), waku, submit=True).status == "rejected"
    late = _plan()
    late.start_ts = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
    assert d.place(late, late.lines[0], submit=True).status == "rejected"
    assert d._page is None                       # どちらもブラウザを起動していない


def test_completion_page_with_error_column_header_is_placed():
    """受付画面にも『エラー内容』の列見出しがある。それで rejected にしてはいけない（実機で発生）。"""
    from narops.orders.rakuten import classify_completion
    body = ("投票完了 当サイトではブラウザの戻るボタンを使用しないようお願い致します。 お客様の投票を以下のように受け付けました。 "
            "受付番号 : 0014 受付時刻 : 16:11 投票件数 : 1 番号 エラー内容 式別 方式 馬／組番 オッズ 金額 概算払戻金 "
            "佐賀 3R 1 - 馬単 通常 2→1 17.5 100円 1,750円 精算可能額 3,590円")
    status, detail, balance = classify_completion(body)
    assert status == "placed" and "0014" in detail and balance == 3590


def test_completion_classifier_rejects_real_errors_and_unknown_pages():
    from narops.orders.rakuten import classify_completion
    assert classify_completion("投票エラー 投票締切時刻を過ぎています")[0] == "rejected"
    assert classify_completion("メンテナンス中です")[0] == "error"


def test_claim_is_atomic_so_two_runners_never_order_the_same_ticket(tmp_path):
    a, b = Ledger(tmp_path / "l.sqlite"), Ledger(tmp_path / "l.sqlite")     # 別プロセス相当の2接続
    assert a.claim("R1", "2026-10-18", "3連単", ((1, 2, 3),), 100, START) is True
    assert b.claim("R1", "2026-10-18", "3連単", ((1, 2, 3),), 100, START) is False
    assert b.claim("R1", "2026-10-18", "馬連", ((1, 2),), 100, START) is True      # 別の券種は別


def test_runner_skips_a_line_another_runner_already_claimed(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    led.claim("212026101805", "2026-10-18", "複勝", ((4,),), 100, START)     # 他のランナーが先に取った
    _, drv, _ = _run(_plan(), _limits(), tmp_path, START - timedelta(minutes=8), ledger=led)
    assert [c[1] for c in drv.calls] == ["3連複"]


def test_pending_counts_toward_the_daily_cap(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    led.claim("other", "2026-10-18", "3連単", ((1, 2, 3),), 900, START)
    assert led.spent("2026-10-18", live=True) == 900 and led.spent("2026-10-18", live=False) == 900
