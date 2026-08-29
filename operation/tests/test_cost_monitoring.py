"""CO-01..06, MO-01..05: コストと監視。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from narops.cost import (
    CostEstimate, ServiceUsage, TARGET_MONTHLY_USD, assert_billing_guard,
    assert_no_always_on, default_usage,
)
from narops.monitoring import (
    ALERT_KINDS, MonitorConfig, check_budget, check_coverage, check_ece,
    check_fetch_failures, check_model_freshness, check_psi, check_rf_guards,
    check_schema_drift, implemented_alert_kinds, should_block_delivery, SLO,
)

pytestmark = pytest.mark.unit


# ------------------------------------------------------------ CO-01（Blocker）
def test_co01_bq_job_requires_bytes_billed():
    assert_billing_guard({"maximum_bytes_billed": 2_000_000_000})
    with pytest.raises(ValueError, match="maximum_bytes_billed"):
        assert_billing_guard({})


# ------------------------------------------------------------------ CO-02
def test_co02_terraform_must_have_min_instances_zero():
    good = {"cloud_run": {"nar-api": {"min_instance_count": 0, "cpu_idle": True,
                                      "region": "asia-northeast1",
                                      "allow_unauthenticated": False}}}
    assert assert_no_always_on(good) == []


def test_co02_detects_always_on_and_wrong_region():
    bad = {"cloud_run": {
        "nar-api": {"min_instance_count": 1, "cpu_idle": False,
                    "region": "us-central1", "allow_unauthenticated": True}}}
    problems = assert_no_always_on(bad)
    assert len(problems) == 4
    assert any("min_instance_count" in p for p in problems)
    assert any("region" in p for p in problems)
    assert any("未認証" in p for p in problems)


def test_co02_real_terraform_config_is_compliant():
    """リポジトリの Terraform 定義そのものを検査する。"""
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "infra" / "services.json"
    cfg = json.loads(p.read_text(encoding="utf-8"))
    assert assert_no_always_on(cfg) == [], "本番定義が常時課金の設定になっています"


# ------------------------------------------------------------------ CO-04
def test_co04_estimated_monthly_cost_within_target():
    est = CostEstimate(default_usage())
    total = est.compute()
    assert total <= TARGET_MONTHLY_USD, (
        f"月額見積 ${total} が目標 ${TARGET_MONTHLY_USD} を超えています: {est.breakdown}")


def test_co04_design_baseline_without_throttling_exceeds_free_tier():
    """設計書 §6.1 の算術を固定する。

    調整前（約500起動/日・平均20秒・1vCPU）は 300,000 vCPU秒/月となり、
    無料枠 240,000 を超えて約 $1.4 が発生する。この事実がオッズ収集を
    締切直前帯に絞る判断の根拠なので、数字ごと回帰テストにしておく。
    """
    before = CostEstimate([ServiceUsage("nar-ops", 500, 20, 1.0, 1.0)], storage_usd=0.0)
    total = before.compute()
    assert before.breakdown["vcpu_seconds"] == pytest.approx(300_000)
    assert total == pytest.approx(1.44, abs=0.01)
    assert total > TARGET_MONTHLY_USD

    after = CostEstimate([ServiceUsage("nar-ops", 300, 20, 1.0, 1.0)], storage_usd=0.0)
    assert after.breakdown is not None
    assert after.compute() == pytest.approx(0.0), "絞った後は無料枠に収まるはず"


def test_co04_breakdown_is_reported():
    est = CostEstimate(default_usage())
    est.compute()
    assert {"vcpu_seconds", "gib_seconds", "vcpu_usd", "storage_usd"} <= set(est.breakdown)


# ------------------------------------------------------------------ MO-01
def test_mo01_ece_alert_boundary():
    """境界の内側で非発火、外側で発火。OOS 比 +0.02 で判定する。"""
    cfg = MonitorConfig(ece_degradation=0.02)
    oos = 0.005
    assert check_ece(pd.Series([oos + 0.019] * 30), oos, cfg) is None, "境界内で発火"
    assert check_ece(pd.Series([oos + 0.020] * 30), oos, cfg) is not None
    assert check_ece(pd.Series([oos + 0.030] * 30), oos, cfg) is not None


def test_mo01_ece_uses_last_30_days_only():
    cfg = MonitorConfig(ece_degradation=0.02)
    # 古い悪化は無視され、直近30日だけが効く
    series = pd.Series([0.5] * 60 + [0.006] * 30)
    assert check_ece(series, 0.005, cfg) is None


def test_mo01_psi_boundary():
    cfg = MonitorConfig(psi_threshold=0.25)
    assert check_psi({"a": 0.24}, cfg) is None
    alert = check_psi({"a": 0.26, "b": 0.10}, cfg)
    assert alert is not None and "a=0.260" in alert.message


def test_mo01_coverage_boundary():
    cfg = MonitorConfig(coverage_min=0.90)
    assert check_coverage(0.91, cfg) is None
    assert check_coverage(0.89, cfg) is not None


def test_mo01_model_freshness_boundary():
    cfg = MonitorConfig(model_stale_days=90)
    assert check_model_freshness(date(2026, 6, 1), date(2026, 8, 25), cfg) is None
    assert check_model_freshness(date(2026, 5, 1), date(2026, 8, 25), cfg) is not None


def test_mo01_fetch_failure_streak():
    cfg = MonitorConfig(fetch_failure_streak=3)
    assert check_fetch_failures(2, cfg) is None
    assert check_fetch_failures(3, cfg) is not None


# ------------------------------------------------------------ RF ガード（本番）
def test_rf_guards_fire_on_impossible_production_results():
    alerts = check_rf_guards(top1_30d=0.72, nll_30d=1.10)
    assert len(alerts) == 2
    assert all(a.severity == "Blocker" for a in alerts)
    assert should_block_delivery(alerts), "RF 発火時は配信を止めるべき"


def test_rf_guard_roi_streak():
    roi = pd.Series([1.35, 1.40, 1.33])
    alerts = check_rf_guards(monthly_roi=roi)
    assert alerts and "RF-03" in alerts[0].message


def test_rf_guards_quiet_on_realistic_numbers():
    assert check_rf_guards(top1_30d=0.44, nll_30d=1.80,
                           monthly_roi=pd.Series([0.92, 1.05, 0.88])) == []


# ------------------------------------------------------------------ MO-02
def test_mo02_all_seven_alert_kinds_are_implemented():
    """設計書 §7.2 の7系統すべてに実装が存在すること。"""
    assert implemented_alert_kinds() == set(ALERT_KINDS)
    assert len(ALERT_KINDS) == 7


def test_mo02_blocking_kinds_stop_delivery():
    from narops.monitoring import Alert

    assert should_block_delivery([Alert("skew_mismatch", "Blocker", "x")])
    assert should_block_delivery([Alert("rf_guard", "Blocker", "x")])
    assert not should_block_delivery([Alert("model_stale", "Critical", "x")])


def test_budget_alert():
    assert check_budget(2.5, threshold=3.0) is None
    assert check_budget(3.1, threshold=3.0) is not None


def test_schema_drift_alert_is_blocker():
    a = check_schema_drift(["entry:37列"])
    assert a is not None and a.severity == "Blocker" and a.blocks_delivery


# ------------------------------------------------------------------ MO-04
def test_mo04_slo_evaluation():
    ok = SLO(notify_in_time=0.96, coverage=0.985, db_fresh_days=0, skew_mismatches=0)
    assert ok.met and ok.violations() == []

    bad = SLO(notify_in_time=0.90, coverage=0.97, db_fresh_days=1, skew_mismatches=2)
    v = bad.violations()
    assert len(v) == 3 and not bad.met


# ------------------------------------------------------------------ MO-05
def test_mo05_structured_log_has_no_secrets_and_is_traceable():
    from narops.config import contains_secret, redact

    webhook = "https://discord.com/api/webhooks/1/xyz"
    entry = {"correlation_id": "202026082511-v1", "event": "infer.done",
             "detail": f"posted to {webhook}"}
    safe = {k: redact(str(v)) for k, v in entry.items()}
    assert not contains_secret(str(safe), [webhook])
    assert safe["correlation_id"] == "202026082511-v1", "相関 ID は残すこと"


# ------------------------------------------------------ 1日あたりの実行回数
def test_daily_limit_survives_a_process_restart(wh, cfg, clock):
    """上限は job_run から数える。

    プロセス内カウンタだけだと、Cloud Run のようにインスタンスが入れ替わる
    環境で上限が効かない（新しいインスタンスは 0 から数え直す）。
    """
    from narops.jobs import UpdateBudget, record_run

    day = clock.now().date()
    first = UpdateBudget(cfg, wh=wh)
    assert first.allow("plan_day", day) is True
    record_run(wh, "plan_day", clock, status="ok")

    # 別インスタンス相当。カウンタは空だが DB には記録がある
    second = UpdateBudget(cfg, wh=wh)
    assert second.allow("plan_day", day) is False


def test_daily_limit_without_a_warehouse_still_works():
    """テスト用にプロセス内カウンタだけでも動くこと。"""
    from datetime import date

    from narops.config import OpsConfig
    from narops.jobs import UpdateBudget

    b = UpdateBudget(OpsConfig.load())
    day = date(2026, 8, 28)
    assert b.allow("plan_day", day) is True
    assert b.allow("plan_day", day) is False
