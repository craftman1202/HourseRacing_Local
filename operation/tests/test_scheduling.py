"""SC-01..07: スケジューリングとタスク展開。"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from narops.clock import FixedClock, jst_datetime, to_jst, to_utc
from narops.scheduling import (
    INFER, SCHEDULER_JOBS, coverage, infer_task_name, notified_in_time, plan_day,
    reconcile, validate_scheduler_jobs,
)
from narops.tasks import AlreadyExists, Task, TaskQueue

pytestmark = pytest.mark.component


@pytest.fixture
def queue() -> TaskQueue:
    return TaskQueue()


# ------------------------------------------------------------------ SC-01
def test_sc01_scheduler_jobs_within_free_tier():
    validate_scheduler_jobs()
    assert len(SCHEDULER_JOBS) <= 3
    assert all(j["tz"] == "Asia/Tokyo" for j in SCHEDULER_JOBS)


def test_sc01_too_many_jobs_is_rejected():
    jobs = SCHEDULER_JOBS + ({"name": "extra", "cron": "* * * * *", "tz": "Asia/Tokyo"},)
    with pytest.raises(ValueError, match="無料枠"):
        validate_scheduler_jobs(jobs)


def test_sc01_non_jst_timezone_is_rejected():
    jobs = ({"name": "a", "cron": "0 0 * * *", "tz": "UTC"},)
    with pytest.raises(ValueError, match="Asia/Tokyo"):
        validate_scheduler_jobs(jobs)


# ------------------------------------------------------------------ SC-02
def test_sc02_one_infer_task_per_race(schedule, queue, clock, cfg):
    plan_day(schedule, queue, clock, cfg)
    assert queue.count(INFER) == len(schedule)


def test_sc02_odds_task_count_within_design_budget(schedule, queue, clock, cfg):
    """1日 100〜120 本以内。起動回数がそのままコストになる。"""
    plan_day(schedule, queue, clock, cfg)
    n_odds = queue.count("/snapshot-odds")
    assert n_odds <= 120, f"オッズ収集タスクが {n_odds} 本で設計本数を超えています"


def test_sc02_infer_tasks_fire_13_minutes_before(schedule, queue, clock, cfg):
    plan_day(schedule, queue, clock, cfg)
    for _, r in schedule.iterrows():
        t = queue.tasks[infer_task_name(r["race_id"], r["start_ts"])]
        assert (to_utc(r["start_ts"]) - t.scheduled_for) == timedelta(minutes=13)


def test_sc02_cancelled_races_are_skipped(schedule, queue, clock, cfg):
    schedule.loc[0, "status"] = "cancelled"
    plan_day(schedule, queue, clock, cfg)
    assert queue.count(INFER) == len(schedule) - 1


def test_sc02_banei_races_are_never_scheduled_for_inference(schedule, queue, clock, cfg):
    """ばんえい（帯広、baba_code 1-4）は確定層に一切入らない（TR-11）ので、
    推論を積んでも毎回 InsufficientData で失敗し、Critical アラートが
    空振りで鳴り続けるだけになる（実際に発生した — 2026-08-29、race_id
    032026082906）。積む前に除外する。
    """
    schedule.loc[0, "baba_code"] = 3
    plan_day(schedule, queue, clock, cfg)
    assert queue.count(INFER) == len(schedule) - 1
    assert queue.tasks.get(infer_task_name(schedule.loc[0, "race_id"],
                                           schedule.loc[0, "start_ts"])) is None


# ------------------------------------------------------------------ SC-03
def test_sc03_reenqueue_of_same_task_is_a_noop(schedule, queue, clock, cfg):
    plan_day(schedule, queue, clock, cfg)
    n = queue.count()
    plan_day(schedule, queue, clock, cfg)          # 再実行
    assert queue.count() == n, "同名タスクが重複しています"


def test_sc03_task_name_is_deterministic():
    ts = jst_datetime(2026, 8, 25, 20, 35)
    assert infer_task_name("202026082511", ts) == infer_task_name("202026082511", ts)
    assert infer_task_name("202026082511", ts) != infer_task_name(
        "202026082511", ts + timedelta(minutes=1))


def test_sc03_strict_enqueue_raises_on_duplicate(queue, clock):
    t = Task("x", INFER, {}, clock.now())
    queue.enqueue_strict(t)
    with pytest.raises(AlreadyExists):
        queue.enqueue_strict(t)


# ------------------------------------------------------------------ SC-04
def test_sc04_delay_beyond_threshold_reschedules(queue, clock, cfg):
    rid = "202026082511"
    old = jst_datetime(2026, 8, 25, 20, 35)
    new = old + timedelta(minutes=20)
    queue.enqueue(Task(infer_task_name(rid, old), INFER, {"race_id": rid},
                       to_utc(old) - timedelta(minutes=13)))

    actions = reconcile({rid: new}, {rid: old}, queue, clock, cfg)
    kinds = [a.kind for a in actions]
    assert "delete" in kinds and "enqueue" in kinds
    assert infer_task_name(rid, old) not in queue.tasks
    assert infer_task_name(rid, new) in queue.tasks
    assert queue.count(INFER) == 1, "二重発火します"


def test_sc04_small_change_does_not_reschedule(queue, clock, cfg):
    rid = "202026082511"
    old = jst_datetime(2026, 8, 25, 20, 35)
    new = old + timedelta(minutes=2)          # 閾値 5 分未満
    queue.enqueue(Task(infer_task_name(rid, old), INFER, {"race_id": rid},
                       to_utc(old) - timedelta(minutes=13)))
    actions = reconcile({rid: new}, {rid: old}, queue, clock, cfg)
    assert actions == []
    assert infer_task_name(rid, old) in queue.tasks


def test_sc04_late_change_switches_to_immediate(cfg):
    """変更が発走13分前を過ぎてから起きたら即時実行に切り替える。"""
    rid = "202026082511"
    old = jst_datetime(2026, 8, 25, 20, 35)
    new = old + timedelta(minutes=6)
    now = FixedClock(new - timedelta(minutes=5))   # もう13分前を過ぎている
    q = TaskQueue()
    q.enqueue(Task(infer_task_name(rid, old), INFER, {"race_id": rid},
                   to_utc(old) - timedelta(minutes=13)))

    actions = reconcile({rid: new}, {rid: old}, q, now, cfg)
    immediate = [a for a in actions if a.kind == "immediate"]
    assert immediate and "遅延" in immediate[0].reason
    assert q.tasks[infer_task_name(rid, new)].payload.get("delayed") is True


def test_sc04_cancelled_race_task_is_deleted(queue, clock, cfg):
    rid = "202026082511"
    old = jst_datetime(2026, 8, 25, 20, 35)
    queue.enqueue(Task(infer_task_name(rid, old), INFER, {"race_id": rid}, to_utc(old)))
    actions = reconcile({}, {rid: old}, queue, clock, cfg)
    assert [a.kind for a in actions] == ["delete"]
    assert queue.count() == 0


def test_sc04_new_race_is_added(queue, clock, cfg):
    rid = "202026082512"
    new = jst_datetime(2026, 8, 25, 21, 5)
    actions = reconcile({rid: new}, {}, queue, clock, cfg)
    assert [a.reason for a in actions] == ["追加開催"]
    assert queue.count(INFER) == 1


# ------------------------------------------------------------------ SC-05
def test_sc05_no_races_means_no_tasks(queue, clock, cfg):
    empty = pd.DataFrame(columns=["race_id", "race_date", "baba_code", "race_no",
                                  "start_ts", "status"])
    actions = plan_day(empty, queue, clock, cfg)
    assert actions == [] and queue.count() == 0, "開催なし日にコストを発生させています"


# ------------------------------------------------------------------ SC-06/07
def test_sc06_coverage_metric():
    assert coverage({"a", "b"}, {"a", "b", "c"}) == pytest.approx(2 / 3)
    assert coverage(set(), set()) == 1.0
    assert coverage({"x"}, {"a"}) == 0.0


def test_sc07_delivery_timeliness():
    start = pd.Timestamp("2026-08-25T11:35:00Z")
    log = pd.DataFrame({
        "sent_at": [start - pd.Timedelta(minutes=12), start - pd.Timedelta(minutes=8)],
        "start_ts": [start, start],
    })
    assert notified_in_time(log, lead_minutes=10) == pytest.approx(0.5)


def test_sc07_empty_log_is_not_a_violation():
    assert notified_in_time(pd.DataFrame(), lead_minutes=10) == 1.0


# ------------------------------------------------------ 予約先の実体（本番）
def test_app_uses_cloud_tasks_when_a_service_url_is_given(monkeypatch):
    """既定のインメモリ実装のままだと、予約がインスタンス終了で消える。

    「/plan-day が N 件登録しました」と報告するのに推論が一度も走らない、
    という気付きにくい壊れ方になる。
    """
    from types import SimpleNamespace

    from narops import app as app_mod
    from narops.config import OpsConfig

    cfg = OpsConfig.load()
    svc = SimpleNamespace()
    monkeypatch.setenv("NAROPS_SERVICE_URL", "https://nar-ops.example.run.app")
    app_mod._attach_queue(svc, cfg)

    assert type(svc.queue).__name__ == "CloudTasksQueue"
    assert svc.queue.service_url.endswith("run.app")
    assert svc.queue.queue == "nar-queue"


def test_app_keeps_the_in_memory_queue_without_a_service_url(monkeypatch):
    from types import SimpleNamespace

    from narops import app as app_mod
    from narops.config import OpsConfig

    svc = SimpleNamespace()
    monkeypatch.delenv("NAROPS_SERVICE_URL", raising=False)
    app_mod._attach_queue(svc, OpsConfig.load())
    assert not hasattr(svc, "queue")


def test_deploy_passes_the_service_url_to_the_inference_service():
    from narops.deploy import build_plan

    plan = build_plan(image="img", mode="paper",
                      service_url="https://nar-ops.example.run.app")
    for step in plan.steps:
        if step.kind != "cloud_run":
            continue
        env = next(a for a in step.command if "NAROPS_ROLE=" in a)
        has_url = "NAROPS_SERVICE_URL=" in env
        assert has_url == (step.name == "nar-ops"), (
            f"{step.name}: サービス URL の有無が役割と合いません")


def test_cloud_tasks_queue_matches_the_in_memory_interface():
    """両実装が同じ入口を持つこと。

    片方に無いメソッドを呼んでいると、本番だけ AttributeError になる
    （実際に /plan-day が `count` で落ちた）。
    """
    from narops.gcp import CloudTasksQueue
    from narops.tasks import TaskQueue

    for name in ("enqueue", "count"):
        assert hasattr(TaskQueue, name) and hasattr(CloudTasksQueue, name), name


def test_plan_day_schedules_the_live_layer_refresh(clock, cfg):
    """`/refresh-live` を誰も予約していなかった。

    エンドポイントも設定値（live_refresh_hours_jst）も揃っていたのに
    plan_day が積んでおらず、当日結果が一度も取り込まれていなかった。
    騎手・調教師の当日成績と馬場差が学習時の粒度で作れなくなる。
    """
    from narops.scheduling import REFRESH_LIVE, plan_day
    from narops.tasks import TaskQueue

    schedule = pd.DataFrame({
        "race_id": ["a", "b"], "status": ["scheduled", "scheduled"],
        "baba_code": [20, 20],
        "start_ts": [to_utc(jst_datetime(2026, 8, 28, 11, 0)),
                     to_utc(jst_datetime(2026, 8, 28, 20, 50))],
    })
    q = TaskQueue()
    plan_day(schedule, q, clock, cfg)

    fired = sorted(to_jst(t.scheduled_for).hour
                   for t in q.tasks.values() if t.endpoint == REFRESH_LIVE)
    assert fired == [12, 16, 20], fired


def test_no_live_refresh_when_nothing_is_running(clock, cfg):
    """開催が無い日に空振りで起動しない。起動回数はそのまま課金になる。"""
    from narops.scheduling import REFRESH_LIVE, plan_day
    from narops.tasks import TaskQueue

    q = TaskQueue()
    plan_day(pd.DataFrame(columns=["race_id", "status", "start_ts"]), q, clock, cfg)
    assert not [t for t in q.tasks.values() if t.endpoint == REFRESH_LIVE]
