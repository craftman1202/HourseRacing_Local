"""日次計画とスケジュール再照合。

Cloud Scheduler は無料枠の3ジョブに厳密に収め、日内の細かい制御は Cloud Tasks で
行う（設計書 §3.1）。発走時刻は実際に変わる（裁決長引き・馬場整備・悪天候）ので、
オッズ収集のたびにスケジュールを再照合して追随する。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Iterator

import pandas as pd

from .clock import JST, Clock, business_date, to_jst, to_utc
from .config import OpsConfig
from .tasks import Task, TaskQueue

INFER = "/infer"
SNAPSHOT = "/snapshot-odds"
REFRESH_LIVE = "/refresh-live"
# ばんえい競馬は別競技（200m 直線・そりの重量が勝敗を決める）で、平地とは
# 別のモデルで推論する。かつては確定層に1行も入れておらず、推論タスクを積むと
# 毎回 InsufficientData で失敗して Critical アラートが空振りしていた
# （2026-08-29、race_id=032026082906）。
# いまは確定層に過去走が入り（refresh.load_history）、ばんえい専用の配布物が
# あれば推論できる。**専用モデルがあるときだけ**積む — 平地モデルで代替すると、
# 距離も回りも無い競技に距離と回りの特徴量で学習したモデルを当てることになる。
BANEI_BABA_CODES = (1, 2, 3, 4)
# Cloud Scheduler ジョブは3個まで（無料枠）。SC-01 の検証対象
SCHEDULER_JOBS = (
    {"name": "ingest-and-refresh", "cron": "40 2 * * *", "tz": "Asia/Tokyo",
     "endpoint": "/ingest-and-refresh"},
    {"name": "plan-day", "cron": "0 8 * * *", "tz": "Asia/Tokyo", "endpoint": "/plan-day"},
    {"name": "weekly-report", "cron": "0 4 * * 1", "tz": "Asia/Tokyo",
     "endpoint": "/weekly-report"},
)
RESCHEDULE_THRESHOLD_SEC = 300


def infer_task_name(race_id: str, start_ts: datetime) -> str:
    """決定的なタスク名。同一発走時刻での重複は Cloud Tasks が自然に弾く。"""
    return f"infer-{race_id}-{int(to_utc(start_ts).timestamp())}"


def snapshot_task_name(race_id: str, at: datetime) -> str:
    return f"odds-{race_id}-{int(to_utc(at).timestamp())}"


@dataclass
class Action:
    kind: str          # enqueue / delete / immediate
    race_id: str
    task_name: str
    scheduled_for: datetime | None = None
    reason: str = ""


def plan_day(schedule: pd.DataFrame, queue: TaskQueue, clock: Clock,
             cfg: OpsConfig, banei_enabled: bool = False) -> list[Action]:
    """当日スケジュールから推論タスクとオッズ収集タスクを積む（SC-02）。

    `banei_enabled` は「ばんえい用の配布物が読み込まれているか」。既定は False で、
    モデルを持たない環境（テスト・平地だけの構成）は従来どおりばんえいを積まない。
    """
    actions: list[Action] = []
    for _, r in schedule.iterrows():
        if str(r.get("status", "scheduled")) != "scheduled":
            continue
        if int(r["baba_code"]) in BANEI_BABA_CODES and not banei_enabled:
            continue
        start = to_utc(r["start_ts"])
        fire = start - timedelta(minutes=cfg.infer_lead_minutes)
        name = infer_task_name(r["race_id"], start)
        if queue.enqueue(Task(name, INFER, {"race_id": r["race_id"]}, fire)):
            actions.append(Action("enqueue", r["race_id"], name, fire, "plan-day"))

    for a in plan_odds_snapshots(schedule, queue, cfg):
        actions.append(a)
    for a in plan_live_refreshes(schedule, queue, cfg):
        actions.append(a)
    return actions


def plan_live_refreshes(schedule: pd.DataFrame, queue: TaskQueue,
                        cfg: OpsConfig) -> list[Action]:
    """ライブ層更新タスク（設計書 §2.2 / §3.1）。

    `/refresh-live` は実装もエンドポイントもあったが、**誰も予約しておらず
    一度も動いていなかった**。当日結果が入らないと騎手・調教師の当日成績と
    馬場差が学習時の粒度で作れず、静かに train-serving skew になる。

    開催が無い日は積まない。空振りの起動はそのまま課金になる。
    """
    actions: list[Action] = []
    if schedule.empty:
        return actions

    starts = [to_utc(t) for t in pd.to_datetime(schedule["start_ts"])]
    first, last = min(starts), max(starts)
    day_jst = to_jst(first).date()

    for hour in cfg.live_refresh_hours:
        fire = to_utc(datetime.combine(day_jst, time(hour=hour),
                                       tzinfo=JST))
        # 初回発走より前・最終発走より後の更新は取るものが無い
        if not (first <= fire <= last + timedelta(minutes=30)):
            continue
        name = f"refresh-live-{int(fire.timestamp())}"
        if queue.enqueue(Task(name, REFRESH_LIVE, {}, fire)):
            actions.append(Action("enqueue", "all", name, fire, "refresh-live"))
    return actions


def plan_odds_snapshots(schedule: pd.DataFrame, queue: TaskQueue,
                        cfg: OpsConfig) -> list[Action]:
    """オッズ収集タスク。

    起動回数がコストに直結するので、締切直前帯だけ高頻度にする。
    オッズの情報価値は締切直前に集中しているので、精度への影響はほぼない
    （設計書 §6.1 の $1 以下維持に必要な調整）。
    """
    actions: list[Action] = []
    if schedule.empty:
        return actions
    starts = [to_utc(t) for t in pd.to_datetime(schedule["start_ts"])]
    first, last = min(starts), max(starts)
    close_window = timedelta(minutes=15)

    t = first - timedelta(minutes=35)
    while t <= last:
        near_close = any(0 <= (s - t).total_seconds() <= close_window.total_seconds()
                         for s in starts)
        step = (cfg.odds_snapshot_close_interval_min if near_close
                else cfg.odds_snapshot_interval_min)
        name = snapshot_task_name("all", t)
        if queue.enqueue(Task(name, SNAPSHOT, {"at": t.isoformat()}, t)):
            actions.append(Action("enqueue", "all", name, t, "odds-snapshot"))
        t += timedelta(minutes=step)
    return actions


def reconcile(current: dict[str, datetime], stored: dict[str, datetime],
              queue: TaskQueue, clock: Clock, cfg: OpsConfig) -> list[Action]:
    """発走時刻変更・追加開催・中止への追随（SC-04）。

    閾値を5分にしているのは、1〜2分の微修正で毎回タスクを作り直すと Tasks 操作が
    無駄に増えるため。ただし変更が発走10分前を過ぎてから発生した場合は
    再エンキューが間に合わないので、即時実行に切り替える。
    """
    actions: list[Action] = []
    for race_id, new_ts in current.items():
        new_ts = to_utc(new_ts)
        old_ts = stored.get(race_id)
        if old_ts is None:
            fire = new_ts - timedelta(minutes=cfg.infer_lead_minutes)
            name = infer_task_name(race_id, new_ts)
            queue.enqueue(Task(name, INFER, {"race_id": race_id}, fire))
            actions.append(Action("enqueue", race_id, name, fire, "追加開催"))
            continue

        old_ts = to_utc(old_ts)
        if abs((new_ts - old_ts).total_seconds()) <= RESCHEDULE_THRESHOLD_SEC:
            continue

        queue.delete(infer_task_name(race_id, old_ts))
        actions.append(Action("delete", race_id, infer_task_name(race_id, old_ts),
                              None, "発走時刻変更"))
        fire = new_ts - timedelta(minutes=cfg.infer_lead_minutes)
        name = infer_task_name(race_id, new_ts)
        if fire <= clock.now():
            queue.enqueue(Task(name, INFER, {"race_id": race_id, "delayed": True},
                               clock.now()))
            actions.append(Action("immediate", race_id, name, clock.now(),
                                  "発走時刻変更により推論が遅延"))
        else:
            queue.enqueue(Task(name, INFER, {"race_id": race_id}, fire))
            actions.append(Action("enqueue", race_id, name, fire, "発走時刻変更"))

    for race_id in set(stored) - set(current):
        name = infer_task_name(race_id, stored[race_id])
        if queue.delete(name):
            actions.append(Action("delete", race_id, name, None, "中止・取消"))
    return actions


def validate_scheduler_jobs(jobs: tuple[dict, ...] = SCHEDULER_JOBS) -> None:
    """SC-01: ジョブ数が3個以下、タイムゾーンが全て Asia/Tokyo。"""
    if len(jobs) > 3:
        raise ValueError(f"Cloud Scheduler ジョブが {len(jobs)} 個で無料枠（3）を超えます")
    bad = [j["name"] for j in jobs if j.get("tz") != "Asia/Tokyo"]
    if bad:
        raise ValueError(f"タイムゾーンが Asia/Tokyo でないジョブ: {bad}")


def coverage(predicted_races: set[str], scheduled_races: set[str]) -> float:
    """推論カバレッジ（SC-06）。分母は投票可能だった出走レース。"""
    if not scheduled_races:
        return 1.0
    return len(predicted_races & scheduled_races) / len(scheduled_races)


def notified_in_time(log: pd.DataFrame, lead_minutes: int = 10) -> float:
    """Discord 到達が発走 N 分前以前だった割合（SC-07）。"""
    if log.empty:
        return 1.0
    sent = pd.to_datetime(log["sent_at"], utc=True)
    start = pd.to_datetime(log["start_ts"], utc=True)
    return float(((start - sent).dt.total_seconds() / 60.0 >= lead_minutes).mean())
