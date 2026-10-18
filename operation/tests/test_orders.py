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


def test_unimplemented_rakuten_driver_never_reports_success():
    from narops.orders.rakuten import RakutenDriver
    assert RakutenDriver().place(_plan(), _plan().lines[0], submit=True).status == "error"
